
import torch
import torch.nn as nn
import torch.sparse as sp
import torch.nn.functional as F
from misc.visualization import save_matimg
from torch_scatter import scatter
import math
from misc.log_utils import log
from ipdb import set_trace

from model.utils import get_mlp


class EdgeUpdate(nn.Module):
    def __init__(self, nhidden, nb_head, edge_type):
        super(EdgeUpdate, self).__init__()
        self.nhidden = nhidden
        self.nb_head = nb_head
        self.edge_type = edge_type

        self.head_mlps = nn.ModuleList([get_mlp(3 * nhidden, nhidden) for _ in range(nb_head)])
        self.final_mlp = get_mlp(nb_head * nhidden, nhidden)

    def forward(self, src_attr, dst_attr, edge_attr):
        combined_input = torch.cat((src_attr, dst_attr, edge_attr), dim=-1)
        
        head_outputs = []
        for head_mlp in self.head_mlps:
            head_output = head_mlp(combined_input)
            head_outputs.append(head_output)
        
        concatenated_heads = torch.cat(head_outputs, dim=-1)
        processed_features = self.final_mlp(concatenated_heads)
        
        edge_feat = edge_attr + processed_features

        return edge_feat


class SimpleEdgeUpdate(nn.Module):
    def __init__(self, nhidden, nout, nb_mlp_layers=2, residual=False, dropout_rate=0.0):
        super(SimpleEdgeUpdate, self).__init__()
        self.nhidden = nhidden
        self.nout = nout
        self.residual = residual

        self.edge_update_mlp = get_mlp(3 * nhidden, nout, nlayer=nb_mlp_layers, dropout_rate=dropout_rate)

    def forward(self, src_attr, dst_attr, edge_attr):
        combined_input = torch.cat((src_attr, dst_attr, edge_attr), dim=-1)
        edge_feat = self.edge_update_mlp(combined_input)

        if self.residual:
            edge_feat = edge_attr + edge_feat

        return edge_feat

class SimpleMPNN(nn.Module):
    def __init__(self, nhidden, nout, nb_mlp_layers=2, residual=False,  bidirectional=False, dropout_rate=0.0):
        super(SimpleMPNN, self).__init__()
        self.nhidden = nhidden
        self.nout = nout
        self.residual = residual
        self.bidirectional = bidirectional
        self.node_aggregate_func = "mean"

        self.flow_temporal_model = get_mlp(3 * nhidden, nhidden, nlayer=nb_mlp_layers, dropout_rate=dropout_rate)
        self.flow_social_model = get_mlp(3 * nhidden, nhidden, nlayer=nb_mlp_layers, dropout_rate=dropout_rate)
        self.flow_view_model = get_mlp(3 * nhidden, nhidden, nlayer=nb_mlp_layers, dropout_rate=dropout_rate)

    def forward(self, node_attr, edge_attr_t, edge_attr_s, edge_attr_v, edge_index_t, edge_index_s, edge_index_v, num_nodes):
        src_t, dst_t = edge_index_t
        src_s, dst_s = edge_index_s
        src_v, dst_v = edge_index_v

        flow_temporal_aggregated = self.compute_msg_from_flow(
            node_attr[src_t], node_attr[dst_t], edge_attr_t, 
            src_t, dst_t, num_nodes,
            flow_model=self.flow_temporal_model,
        )

        flow_social_aggregated = self.compute_msg_from_flow(
            node_attr[src_s], node_attr[dst_s], edge_attr_s, 
            src_s, dst_s, num_nodes,
            flow_model=self.flow_social_model,
        )

        flow_view_aggregated = self.compute_msg_from_flow(
            node_attr[src_v], node_attr[dst_v], edge_attr_v, 
            src_v, dst_v, num_nodes,
            flow_model=self.flow_view_model,
        )

        node_update = flow_temporal_aggregated + flow_social_aggregated + flow_view_aggregated

        if self.residual:
            node_update = node_attr + node_update

        return node_update


    def compute_msg_from_flow(self, src_attr, dst_attr, edge_attr, src_nodes, dst_nodes, dim_size, flow_model):
        
        flow_input = torch.cat([src_attr, dst_attr, edge_attr], dim=1)
        flow = flow_model(flow_input)
        
        if self.bidirectional:
            flow_aggregated = scatter(torch.cat((flow, flow), dim=0),
                                      torch.cat((src_nodes, dst_nodes)),
                                      dim=0,
                                      reduce=self.node_aggregate_func,
                                      dim_size=dim_size)
        else:
            flow_aggregated = scatter(flow, dst_nodes, 
                                      dim=0, 
                                      reduce=self.node_aggregate_func, 
                                      dim_size=dim_size)
        
        return flow_aggregated


class SimpleCameraMPNN(nn.Module):
    def __init__(self, nhidden, nout, nb_mlp_layers=2, residual=False,  bidirectional=False, dropout_rate=0.0):
        super(SimpleCameraMPNN, self).__init__()
        self.nhidden = nhidden
        self.nout = nout
        self.residual = residual
        self.bidirectional = bidirectional
        self.node_aggregate_func = "mean"

        self.flow_temporal_model = get_mlp(3 * nhidden, nhidden, nlayer=nb_mlp_layers, dropout_rate=dropout_rate)
        self.flow_social_model = get_mlp(3 * nhidden, nhidden, nlayer=nb_mlp_layers, dropout_rate=dropout_rate)
        self.flow_view_model = get_mlp(3 * nhidden, nhidden, nlayer=nb_mlp_layers, dropout_rate=dropout_rate)
        self.flow_next_model = get_mlp(3 * nhidden, nhidden, nlayer=nb_mlp_layers, dropout_rate=dropout_rate)
        self.flow_see_model = get_mlp(3 * nhidden, nhidden, nlayer=nb_mlp_layers, dropout_rate=dropout_rate)

    def forward(self, node_attr, edge_attr_t, edge_attr_s, edge_attr_v, camera_node_attr, camera_see_edge_attr, camera_next_edge_attr, edge_index_t, edge_index_s, edge_index_v, edge_index_see, edge_index_next, num_nodes, num_cameras):
        src_t, dst_t = edge_index_t
        src_s, dst_s = edge_index_s
        src_v, dst_v = edge_index_v
        if edge_index_see is not None:  
            src_see, dst_see = edge_index_see
        if edge_index_next is not None:
            src_next, dst_next = edge_index_next


        flow_temporal_aggregated = self.compute_msg_from_flow(
            node_attr[src_t], node_attr[dst_t], edge_attr_t, 
            src_t, dst_t, num_nodes,
            flow_model=self.flow_temporal_model,
        )

        flow_social_aggregated = self.compute_msg_from_flow(
            node_attr[src_s], node_attr[dst_s], edge_attr_s, 
            src_s, dst_s, num_nodes,
            flow_model=self.flow_social_model,
        )

        flow_view_aggregated = self.compute_msg_from_flow(
            node_attr[src_v], node_attr[dst_v], edge_attr_v, 
            src_v, dst_v, num_nodes,
            flow_model=self.flow_view_model,
        )
        
        if edge_index_next is not None:
            flow_next_aggregated = self.compute_msg_from_flow(
                camera_node_attr[src_next], camera_node_attr[dst_next], camera_next_edge_attr, 
                src_next, dst_next, num_cameras,
                flow_model=self.flow_next_model,
            )
        else:
            flow_next_aggregated = torch.zeros_like(camera_node_attr)

        if edge_index_see is not None:
            flow_see_aggregated_node = self.compute_msg_from_flow(
                camera_node_attr[src_see], node_attr[dst_see], camera_see_edge_attr, 
                src_see, dst_see, num_nodes,
                flow_model=self.flow_see_model,
                force_unidirectional=True
            )
        else:
            flow_see_aggregated_node = torch.zeros_like(flow_temporal_aggregated)

        if edge_index_see is not None:
            flow_see_aggregated_camera = self.compute_msg_from_flow(
                node_attr[dst_see], camera_node_attr[src_see], camera_see_edge_attr, 
                dst_see, src_see, num_cameras,
                flow_model=self.flow_see_model,
                force_unidirectional=True,
            )
        else:
            flow_see_aggregated_camera = torch.zeros_like(camera_node_attr)

        camera_node_update = flow_next_aggregated + flow_see_aggregated_camera
        node_update = flow_temporal_aggregated + flow_social_aggregated + flow_view_aggregated + flow_see_aggregated_node

        if self.residual:
            node_update = node_attr + node_update
            camera_node_update = camera_node_attr + camera_node_update

        return node_update, camera_node_update


    def compute_msg_from_flow(self, src_attr, dst_attr, edge_attr, src_nodes, dst_nodes, dim_size, flow_model, force_unidirectional=False):
        
        flow_input = torch.cat([src_attr, dst_attr, edge_attr], dim=1)
        flow = flow_model(flow_input)
        
        if self.bidirectional and not force_unidirectional:
            flow_aggregated = scatter(torch.cat((flow, flow), dim=0),
                                      torch.cat((src_nodes, dst_nodes)),
                                      dim=0,
                                      reduce=self.node_aggregate_func,
                                      dim_size=dim_size)
        else:
            flow_aggregated = scatter(flow, dst_nodes, 
                                      dim=0, 
                                      reduce=self.node_aggregate_func, 
                                      dim_size=dim_size)
        
        return flow_aggregated


# class SelfAttentionAggregatorFull(nn.Module):
#     def __init__(self, nhidden, nout=None, num_heads=4, indice_out=0 ,residual=True):
#         super(SelfAttentionAggregatorFull, self).__init__()
#         self.residual = residual
#         self.indice_out = indice_out
#         self.attention = nn.MultiheadAttention(embed_dim=nhidden, num_heads=num_heads, batch_first=True)

#         if nout is None:
#             nout = nhidden
            
#         self.fc = nn.Linear(nhidden, nout)  # Optional projection after attention

#     def forward(self, triplets):
#         """
#         Aggregates the triplets using self-attention over the nhidden dimension.
#         triplets: Tensor of shape (B, 3, nhidden)
#         """
#         # Transpose to match the MultiheadAttention input (sequence_len, batch_size, embedding_dim)
#         # Apply multi-head attention (attends to the nhidden dimension)
#         attention_output, _ = self.attention(triplets, triplets, triplets)

#         # Optionally apply a linear transformation to the attention output
#         aggregated = self.fc(attention_output[:, self.indice_out])
        
#         # Update one of the triplet elements (e.g., first element) and optionally use a residual connection
#         if self.residual:
#             updated_triplet = aggregated + triplets[:, self.indice_out, :]  # Residual connection with first vector
#         else:
#             updated_triplet = aggregated

#         return updated_triplet


# class SimpleEdgeUpdate(nn.Module):
#     def __init__(self, nhidden, residual=False):
#         super(SimpleEdgeUpdate, self).__init__()
#         self.nhidden = nhidden
#         self.residual = residual

#         self.edge_update_op = SelfAttentionAggregatorFull(nhidden, residual=residual)

#     def forward(self, src_attr, dst_attr, edge_attr):
#         combined_input = torch.stack((edge_attr, src_attr, dst_attr), dim=1)
#         edge_feat = self.edge_update_op(combined_input)

#         return edge_feat


# class SimpleMPNN(nn.Module):
#     def __init__(self, nhidden, nout, residual=False, bidirectional=True):
#         super(SimpleMPNN, self).__init__()
#         self.nhidden = nhidden
#         self.nout = nout
#         self.residual = residual
#         self.bidirectional = bidirectional
        
#         self.node_aggregate_func = "mean"

#         indice_out = slice(0,2) if self.bidirectional else 0
#         self.flow_temporal_model = SelfAttentionAggregatorFull(nhidden, num_heads=1, indice_out=indice_out)#SAGEConv(in_channels=3 * nhidden, out_channels=nhidden)
#         self.flow_social_model = SelfAttentionAggregatorFull(nhidden, num_heads=1, indice_out=indice_out)#SAGEConv(in_channels=3 * nhidden, out_channels=nhidden)
#         self.flow_view_model = SelfAttentionAggregatorFull(nhidden, num_heads=1, indice_out=indice_out)#SAGEConv(in_channels=3 * nhidden, out_channels=nhidden)

#     def forward(self, node_attr, edge_attr_t, edge_attr_s, edge_attr_v, edge_index_t, edge_index_s, edge_index_v, num_nodes):
#         src_t, dst_t = edge_index_t
#         src_s, dst_s = edge_index_s
#         src_v, dst_v = edge_index_v

#         flow_temporal_aggregated = self.compute_msg_from_flow(
#             node_attr[src_t], node_attr[dst_t], edge_attr_t, 
#             src_t, dst_t, num_nodes,
#             flow_model=self.flow_temporal_model
#         )

#         flow_social_aggregated = self.compute_msg_from_flow(
#             node_attr[src_s], node_attr[dst_s], edge_attr_s, 
#             src_s, dst_s, num_nodes,
#             flow_model=self.flow_social_model
#         )

#         flow_view_aggregated = self.compute_msg_from_flow(
#             node_attr[src_v], node_attr[dst_v], edge_attr_v, 
#             src_v, dst_v, num_nodes,
#             flow_model=self.flow_view_model
#         )

#         node_update = flow_temporal_aggregated + flow_social_aggregated + flow_view_aggregated

#         if self.residual:
#             node_update = node_attr + node_update

#         return node_update


    # def compute_msg_from_flow(self, src_attr, dst_attr, edge_attr, src_nodes, dst_nodes, dim_size, flow_model):
        
    #     flow_input = torch.stack([src_attr, dst_attr, edge_attr], dim=1)
    #     flow = flow_model(flow_input)
        
    #     if self.bidirectional:
    #         flow = flow.reshape(flow.shape[0]*flow.shape[1], flow.shape[-1]) 
    #         flow_aggregated = scatter(flow,
    #                                   torch.cat((src_nodes, dst_nodes)),
    #                                   dim=0,
    #                                   reduce=self.node_aggregate_func,
    #                                   dim_size=dim_size)
    #     else:
    #         flow_aggregated = scatter(flow, dst_nodes, 
    #                                   dim=0, 
    #                                   reduce=self.node_aggregate_func, 
    #                                   dim_size=dim_size)
        
    #     return flow_aggregated

class FactorMPNN(nn.Module):
    def __init__(self, nhidden, nout):
        super(FactorMPNN, self).__init__()
        self.nhidden = nhidden

        self.node_aggregate_func = "mean"

        self.flow_forward_model = get_mlp(3 * nhidden, nhidden)
        self.flow_backward_model = get_mlp(3 * nhidden, nhidden)

        self.flow_social_model = get_mlp(3 * nhidden, nhidden)
        self.flow_view_model = get_mlp(3 * nhidden, nhidden)

        self.total_flow_model = get_mlp(4 * nhidden, nout)


    def forward(self, node_attr, edge_attr_t, edge_attr_s, edge_attr_v, edge_index_t, edge_index_s, edge_index_v, num_nodes):
        src_t, dst_t = edge_index_t
        src_s, dst_s = edge_index_s
        src_v, dst_v = edge_index_v

        # Compute forward flow
        flow_forward_aggregated = self.compute_msg_from_flow(
            node_attr[src_t], node_attr[dst_t], edge_attr_t, 
            src_t, dst_t, num_nodes,
            flow_model=self.flow_forward_model
        )

        # Compute backward flow
        flow_backward_aggregated = self.compute_msg_from_flow(
            node_attr[dst_t], node_attr[src_t], edge_attr_t, 
            dst_t, src_t, num_nodes,
            flow_model=self.flow_backward_model
        )

        # Compute frame flow
        flow_frame_aggregated = self.compute_msg_from_flow(
            node_attr[src_s], node_attr[dst_s], edge_attr_s, 
            src_s, dst_s, num_nodes,
            flow_model=self.flow_social_model,
            bidirectional=True
        )

        # Compute view flow
        flow_view_aggregated = self.compute_msg_from_flow(
            node_attr[src_v], node_attr[dst_v], edge_attr_v, 
            src_v, dst_v, num_nodes,
            flow_model=self.flow_view_model,
            bidirectional=True
        )

        pyg_msgagg = torch.cat((flow_forward_aggregated, flow_backward_aggregated, flow_frame_aggregated, flow_view_aggregated), dim=1)

        node_update = self.total_flow_model(pyg_msgagg)

        return node_update
    
    def compute_msg_from_flow(self, src_attr, dst_attr, edge_attr, src_nodes, dst_nodes, dim_size, flow_model, bidirectional=False):
        """
        Compute the message from the flow.

        Args:
            src_attr (torch.Tensor): Source node attributes.
            dst_attr (torch.Tensor): Destination node attributes.
            edge_attr (torch.Tensor): Edge attributes.
            src_nodes (torch.Tensor): Indices of source nodes.
            dst_nodes (torch.Tensor): Indices of destination nodes.
            dim_size (int): The size of the output tensor, typically equal to the total number of nodes in the graph.
                            This ensures that the output tensor has enough space for all nodes, even if some nodes
                            don't receive any messages.
            flow_model (nn.Module): The flow model to use for computing messages.
            bidirectional (bool): Flag to switch between unidirectional and bidirectional flow. Default is False.

        Returns:
            torch.Tensor: Aggregated flow messages for each node.

        This function concatenates source, destination, and edge attributes,
        passes them through a flow model, and then aggregates the resulting
        messages for each node using scatter operation. It can handle both
        unidirectional and bidirectional flows based on the bidirectional flag.
        """
        
        flow_input = torch.cat([src_attr, dst_attr, edge_attr], dim=1)
        flow = flow_model(flow_input)
        
        if bidirectional:
            flow_aggregated = scatter(torch.cat((flow, flow), dim=0),
                                      torch.cat((src_nodes, dst_nodes)),
                                      dim=0,
                                      reduce=self.node_aggregate_func,
                                      dim_size=dim_size)
        else:
            flow_aggregated = scatter(flow, dst_nodes, 
                                      dim=0, 
                                      reduce=self.node_aggregate_func, 
                                      dim_size=dim_size)
        
        return flow_aggregated

        # # Learnable parameters
        # self.W_q = nn.Linear(nhidden, nhidden)
        # self.W_k = nn.Linear(nhidden, nhidden)
        # self.W_v = nn.Linear(nhidden, nhidden)
        # self.W_o = nn.Linear(nhidden, nhidden)

    # def compute_msg_from_flow_with_attention(self, src_attr, dst_attr, edge_attr, src_nodes, dst_nodes, dim_size, flow_model, bidirectional=False):
    #     """
    #     Compute the message from the flow using multi-headed attention.

    #     Args:
    #         src_attr (torch.Tensor): Source node attributes.
    #         dst_attr (torch.Tensor): Destination node attributes.
    #         edge_attr (torch.Tensor): Edge attributes.
    #         src_nodes (torch.Tensor): Indices of source nodes.
    #         dst_nodes (torch.Tensor): Indices of destination nodes.
    #         dim_size (int): The size of the output tensor.
    #         flow_model (nn.Module): The flow model to use for computing messages.
    #         bidirectional (bool): Flag to switch between unidirectional and bidirectional flow. Default is False.

    #     Returns:
    #         torch.Tensor: Aggregated flow messages for each node.
    #     """
        
    #     if src_attr.size(0) == 0:
    #         # Handle empty input case
    #         return torch.zeros((dim_size, self.nhidden), device=src_attr.device)
        
    #     flow_input = torch.cat([src_attr, dst_attr, edge_attr], dim=1)
    #     flow = flow_model(flow_input)
        
    #     # Define multi-head attention parameters
    #     num_heads = 4
    #     d_model = flow.size(1)
    #     d_k = d_model // num_heads
        
    #     # Compute Q, K, V
    #     Q = self.W_q(flow).view(-1, num_heads, d_k)
    #     K = self.W_k(flow).view(-1, num_heads, d_k)
    #     V = self.W_v(flow).view(-1, num_heads, d_k)
        
    #     # Compute attention scores
    #     scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        
    #     # Apply softmax to get attention weights
    #     attn_weights = F.softmax(scores, dim=-1)
        
    #     # Apply attention weights
    #     flow_aggregated = torch.matmul(attn_weights, V)
        
    #     # Reshape and apply output transformation
    #     flow_aggregated = flow_aggregated.view(-1, d_model)
    #     flow_aggregated = self.W_o(flow_aggregated)
        
    #     # Aggregate messages for each node
    #     if bidirectional:
    #         nodes = torch.cat((src_nodes, dst_nodes))
    #         flow_aggregated = torch.cat((flow_aggregated, flow_aggregated), dim=0)
    #     else:
    #         nodes = dst_nodes
        
    #     flow_aggregated = scatter(flow_aggregated, nodes, 
    #                               dim=0, 
    #                               reduce=self.node_aggregate_func, 
    #                               dim_size=dim_size)
        
    #     return flow_aggregated



    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.nhidden)
            + ", "
            + str(self.nhidden)
            + ")"
        )