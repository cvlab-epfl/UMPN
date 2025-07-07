from ipdb import set_trace
from torch_geometric.data import Data
import torch
from typing import List, Tuple, Dict,Set
import networkx as nx
from configs.constants import SEQ_INDICES_MAPPING, OFFSET_ATTR, CUMULATE_ATTR
import numpy as np
from loguru import logger

def extract_sequence_from_graph(graph: Data, index: int) -> Data:
    """
    Extracts a single sequence (single view) from a multi view graph based on the given index.

    Args:
        graph (Data): The graph object to extract the sequence from.
        index (int): The index of the sequence to extract.

    Returns:
        Data: A new graph object containing only the data for the specified sequence.
    """
    
    # Extract the sequences
    seq_indices_edge = graph.seq_indices_edge
    seq_indices_node = graph.seq_indices_node
    
    # Identify the positions of the sequence
    edge_indices = (seq_indices_edge == index).nonzero().squeeze()
    node_indices = (seq_indices_node == index).nonzero().squeeze()

    offset = node_indices[0]
    # Extract the relevant data for that sequence
    edge_index = graph.edge_index[:, edge_indices] - offset
    labels_e = graph.labels_e[edge_indices]
    if len(edge_index.shape) == 1:
        edge_index = edge_index.unsqueeze(0).T # Make sure edge_index is of shape (2, num_edges)
        labels_e = graph.labels_e[edge_indices].unsqueeze(0) # Make sure labels_e is of shape (num_edges, )
    
    # Create a new graph structure for the single sequence
    return Data(
        x=graph.x[node_indices],
        edge_index=edge_index,
        labels_n=graph.labels_n[node_indices],
        labels_e=labels_e,
        y_pred=graph.y_pred[node_indices],
        edge_attr=graph.edge_attr[edge_indices],
        labels_view=graph.labels_view[node_indices],
        seq_indices_edge=seq_indices_edge[edge_indices],
        seq_indices_node=seq_indices_node[node_indices]
    )

def extract_sequence_from_model_output(graph: Data, index: int) -> Data:

    # Extract the sequences
    seq_indices_edge = graph.seq_indices_edge
    seq_indices_node = graph.seq_indices_node
    
    # Identify the positions of the sequence
    edge_indices = (seq_indices_edge == index).nonzero().squeeze()
    node_indices = (seq_indices_node == index).nonzero().squeeze()

    return Data(
        h_out_n=graph.h_out_n[node_indices],
        h_out_e=graph.h_out_e[edge_indices],
        y_n_act=graph.y_n_act[node_indices],
        y_e_act=graph.y_e_act[edge_indices],
        y_n=graph.y_n[node_indices],
        y_e=graph.y_e[edge_indices],
        seq_indices_edge=seq_indices_edge[edge_indices],
        seq_indices_node=seq_indices_node[node_indices]
    )


from torch_geometric.data import Data
import torch

def extract_sequence(graph: Data, index: int) -> Data:
    """
    Extracts a single sequence (single view) from a graph based on the given index.

    Args:
        graph (Data): The graph object to extract the sequence from.
        index (int): The index of the sequence to extract.

    Returns:
        Data: A new graph object containing only the data for the specified sequence.
    """

    graph_attributes = graph.keys()
    extracted_data_dict = {}
    for sim_attr, attr_list in SEQ_INDICES_MAPPING.items():
        if not sim_attr in graph_attributes:
                continue
        indices = (getattr(graph, sim_attr) == index).nonzero().squeeze().reshape(-1)
        
        if OFFSET_ATTR in graph_attributes:
            offset = (getattr(graph, OFFSET_ATTR) == index).nonzero().squeeze().reshape(-1)[0]

        for attr in attr_list:
            if not attr in graph_attributes:
                continue
            
            try:
                if hasattr(graph, attr):
                    if getattr(graph, attr) is None:
                        continue

                    
                    # Additional processing for 'edge_index'
                    if attr == 'edge_index':
                        #set_trace()
                        extracted_data_dict[attr] = getattr(graph, attr)[:, indices] - offset
                        # Ensure correct shape
                        if len(extracted_data_dict[attr].shape) == 1:
                            extracted_data_dict[attr] = extracted_data_dict[attr].unsqueeze(0).T
                    elif attr == 'y_pred_idx':
                        extracted_data_dict[attr] = getattr(graph, attr)[indices]
                        notset = extracted_data_dict[attr]==-1
                        extracted_data_dict[attr] -= offset
                        extracted_data_dict[attr][notset] = -1
                    elif attr == 'ccs':
                        extracted_data_dict[attr] = getattr(graph, attr)[indices]
                        extracted_data_dict[attr][:,0] -= offset
                    elif attr == 'y_out':
                        #set_trace()
                        offset_traj = indices[0].item()
                        extracted_data_dict[attr] = getattr(graph, attr)[indices]
                        notset = extracted_data_dict[attr][:,1]==-1
                        extracted_data_dict[attr][:,1] -= offset_traj
                        extracted_data_dict[attr][notset,1] = -1
                    else:
                        extracted_data_dict[attr] = getattr(graph, attr)[indices]
            except:
                set_trace()
    return Data(**extracted_data_dict)

def concatenate_graphs(graphs: List[Data]) -> Data:
    """
    Concatenates a list of graphs (of different views) into a single multi-view graph.
    
    Args:
        graphs (List[Data]): A list of graphs to concatenate.
        
    Returns:
        Data: A single graph containing the concatenated data attributes.
    """
    
    # Determine attributes present in the Data objects
    try:
        attributes = graphs[0].keys()
    except:
        set_trace()
    concatenated_data_dict = {}
    
    cumulate = False
    num_nodes_cumulative = 0

    trajids_cumulative = 0
    for graph in graphs:
        # Adjust edge_index based on accumulated number of nodes
        if 'edge_index' in attributes:
            cumulate=True
            if 'edge_index' not in concatenated_data_dict:
                concatenated_data_dict['edge_index'] = []
            concatenated_data_dict['edge_index'].append(graph.edge_index + num_nodes_cumulative)
        if "y_pred_idx" in attributes:
            #set_trace()
            cumulate=True
            if "y_pred_idx" not in concatenated_data_dict:
                concatenated_data_dict["y_pred_idx"] = []
            y_pred = graph.y_pred_idx.clone()
            notset = y_pred==-1
            y_pred += num_nodes_cumulative
            y_pred[notset] = -1
            concatenated_data_dict["y_pred_idx"].append(y_pred.cuda())
        if "ccs" in attributes:
            cumulate=True
            if "ccs" not in concatenated_data_dict:
                concatenated_data_dict["ccs"] = []
            ccs = graph.ccs.clone()
            ccs[:,0] += num_nodes_cumulative
            concatenated_data_dict["ccs"].append(ccs.cuda())
        if "y_out" in attributes:
            #set_trace()
            if "y_out" not in concatenated_data_dict:
                concatenated_data_dict["y_out"] = []
            y_out = graph.y_out.copy()
            notset = y_out[:,1]==-1
            y_out[:,1] += trajids_cumulative
            y_out[notset,1] = -1
            trajids_cumulative += graph.y_out.shape[0]
            concatenated_data_dict["y_out"].append(y_out)
        if cumulate:
            if CUMULATE_ATTR in attributes:
                num_nodes_cumulative += graph.labels_n.size(0)
            else:
                raise ValueError("Node number accumulator not found: labels_n not in attributes while updating edge_index.")
        
        # Concatenate other attributes

        for attr in attributes:
            if attr == 'edge_index':  # edge_index already handled
                continue

            if attr == 'y_pred_idx':  # y_pred already handled
                continue
            if attr == 'y_out':  # y_pred already handled
                continue

            if attr not in concatenated_data_dict:
                concatenated_data_dict[attr] = []
            #set_trace()
            concatenated_data_dict[attr].append(getattr(graph, attr))
    
    # Convert lists in concatenated_data_dict to tensors
    for attr, value_list in concatenated_data_dict.items():
        try:
            if attr == 'edge_index':
                concatenated_data_dict[attr] = torch.cat(value_list, dim=1)
            else:
                try:
                    concatenated_data_dict[attr] = torch.cat(value_list, 0)
                except TypeError:
                    concatenated_data_dict[attr] = np.concatenate(value_list, 0)
        except:
            set_trace()
    return Data(**concatenated_data_dict)


def update_seq_indices_attrs(graph: Data, seq:int) -> None:
    """
    Update the sequence indices attributes of a given graph with a new sequence index.
    
    Args:
    - graph (Data): A PyTorch Geometric Data object.
    - seq (int): The new sequence index to be assigned to the sequence indices attributes.

    Returns:
    - None
    """
    for sim_attr, attr_list in SEQ_INDICES_MAPPING.items():
        for attr in attr_list:
            if attr == sim_attr:
                continue
            if attr == 'edge_index': # edge_index.shape (2, num_edges)
                continue
            if attr == 'y_pred_idx':
                continue
            if attr in graph.keys():
                new_seq_indices = torch.full(
                    (getattr(graph, attr).shape[0],), fill_value=seq, dtype=torch.uint8
                )

                # if sim_attr in graph.keys:
                #     logger.debug(f"Updated {sim_attr} [{getattr(graph, sim_attr).shape} -> {new_seq_indices.shape}] with new sequence indices using {attr}.shape")
                # else:
                #     logger.debug(f"Created {sim_attr} [{new_seq_indices.shape}] with new sequence indices using {attr}.shape")
                
                setattr(graph, sim_attr, new_seq_indices)
                break
    

def check_shapes_match(data: Data, verbose=False) -> Tuple[bool, str]:
    for key, attributes in SEQ_INDICES_MAPPING.items():
        for attribute in attributes:
            if attribute in data:
                if attribute == "edge_index":
                    if verbose:
                        print(f"{attribute}: {data[attribute].shape} | {key}: {data[key].shape}")
                    if data[attribute].shape[1] != data[key].shape[0]:
                        return False, f"{attribute} shape does not match with {key}"
                else:
                    if verbose:
                            print(f"{attribute}: {data[attribute].shape} | {key}: {data[key].shape}")
                    if data[attribute].shape[0] != data[key].shape[0]:
                        return False, f"{attribute} shape does not match with {key}"
    return True, "All shapes match"

def check_shapes_match_between_data(data1: Data, data2: Data, verbose=False) -> Tuple[bool, str]:
    for attribute in SEQ_INDICES_MAPPING:
        if verbose:
            print(f"Checking {attribute}... ",end = "")
        if attribute in data1.keys() and attribute in data2.keys():
            if verbose:
                print(f"{attribute}: {getattr(data1,attribute).shape} | {attribute}: {getattr(data2,attribute).shape}")
            if getattr(data1,attribute).shape[0] != getattr(data2,attribute).shape[0]:
                return False,f"Shape mismatch for {attribute}: data1 has {getattr(data1,attribute).shape[0]} and data2 has {getattr(data2,attribute).shape[0]}"
            if verbose:
                print("OK")
        else:
            if verbose:
                print("NOT FOUND")
    return True, "Shared shapes between data1 and data2 match"


def check_data_match(data1: Data, data2: Data, verbose=False) -> Tuple[bool, str]:
    return_bool = True
    for attribute in data1.keys():
        if verbose:
            print(f"Checking {attribute}... ",end = "")
        if attribute in data2.keys():
            if verbose:
                print(f"{attribute}: {getattr(data1,attribute).shape} | {attribute}: {getattr(data2,attribute).shape} ::: ",end = "")
            
            if isinstance(getattr(data1,attribute), torch.Tensor):
                if not getattr(data1,attribute).equal(getattr(data2,attribute)):
                    print(f"Data mismatch for {attribute}")
                    return_bool = False
            else:
                if not (getattr(data1,attribute) == (getattr(data2,attribute))).all():
                        print(f"Data mismatch for {attribute}")
                        return_bool = False
            
            if verbose:
                print("OK")
        else:
            if verbose:
                print("NOT FOUND in data2")
    print("Shared data between data1 and data2 match")
    return return_bool

def create_view_gt_edges(data:Data, timepoint=None) -> None:
    #set_trace()
    if "labels_view" not in data.keys():
        raise ValueError("data1 does not contain labels_view")
    node_timestamps = data.labels_view[:,0]
    unique_timestamps = torch.unique(node_timestamps)

    if len(unique_timestamps) == 0:
        data.edge_index_v = torch.tensor([[]], dtype=torch.long)
        data.y_e_v_gt = torch.tensor([[]], dtype=torch.long)
        return
    gt_view_edge_list = []

    view_edge_list = []
    for t in unique_timestamps:
        # xmask = node_timestamps == t
        # labels_views_t = data.labels_view[xmask]
        # cam_views_t = data.seq_indices_node[xmask]
        
        # object_ids = labels_views_t[:,1]

        # if len(object_ids) == 1:
        #     continue
        
        # adj_mat_gt = torch.tile(object_ids, (len(object_ids), 1))
        # adj_mat_gt = adj_mat_gt == adj_mat_gt.T
        # adj_mat_gt[torch.eye(len(object_ids)).bool()] = False
        # adj_mat_gt = adj_mat_gt.to_sparse()._indices() #indexing starts from 0 here
        # #object_ids[adj_mat_gt].T this should connect now same object ids
        # og_indices = xmask.nonzero().reshape(-1) #original indices
        # try:
        #     adj_mat_gt = og_indices[adj_mat_gt] #get back original indices
        # except:
        #     logger.debug(f"no edges for timestamp {t}")
        #     #continue #no edges for this timestamp
        #     set_trace()
        # #pyg_data.labels_view[:,1][adj_mat_gt].T still should yield same result as above
        # gt_view_edge_list.append(adj_mat_gt)

        # adj_mat = torch.ones((len(object_ids),len(object_ids))).bool()

        # start = 0
        # change_points = np.where(np.diff(cam_views_t))[0] + 1
        # for end in change_points:
        #     adj_mat[start:end, start:end] = False
        #     start = end
        # adj_mat[start:, start:] = False
        # adj_mat = adj_mat.to_sparse()._indices()
        # adj_mat = og_indices[adj_mat]

        # view_edge_list.append(adj_mat)

        edge_index_v, edge_index_v_gt = get_view_edges_for_timepoint(data, t)
        if edge_index_v is None:
            continue
        view_edge_list.append(edge_index_v)
        gt_view_edge_list.append(edge_index_v_gt)

    data.y_e_v_gt = torch.cat(gt_view_edge_list, dim=1)
    data.edge_index_v = torch.cat(view_edge_list, dim=1)

    if timepoint is not None:
        #set_trace()
        edge_index_v, _ = get_view_edges_for_timepoint(data, timepoint)
        data.edge_attr_v = torch.zeros((edge_index_v.shape[1],data.edge_attr.shape[1]), dtype=torch.long).cuda()

def get_view_edges_for_timepoint(data:Data, timepoint:int, view_edge_radius:int=100):
    node_timestamps = data.labels_view[:,0]
    xmask = node_timestamps == timepoint
    labels_views_t = data.labels_view[xmask.cpu()]
    cam_views_t = data.seq_indices_node[xmask.cpu()]

    # import pdb; pdb.set_trace()

    # data_x_t = data.x[xmask.cpu()]

    
    object_ids = labels_views_t[:,1]

    if len(object_ids) == 1:
        None,None
    
    adj_mat_gt = torch.tile(object_ids, (len(object_ids), 1))
    adj_mat_gt = adj_mat_gt == adj_mat_gt.T
    adj_mat_gt[torch.eye(len(object_ids)).bool()] = False
    adj_mat_gt = adj_mat_gt.to_sparse()._indices() #indexing starts from 0 here
    #object_ids[adj_mat_gt].T this should connect now same object ids
    og_indices = xmask.nonzero().reshape(-1) #original indices
    try:
        adj_mat_gt = og_indices[adj_mat_gt] #get back original indices
    except:
        logger.debug(f"no edges for timestamp {timepoint}")
        #continue #no edges for this timestamp
        set_trace()
    #pyg_data.labels_view[:,1][adj_mat_gt].T still should yield same result as above
    #gt_view_edge_list.append(adj_mat_gt)

    # if spatial feature are present only create edge in neihboring region

    #TODO: find a better way to detect if spatial feature are used
    if False and data.x.shape[1] == 11:
        # logger.warning("Using world coordinate to restrict potential view edges")
        adj_mat = torch.zeros((len(object_ids),len(object_ids))).bool()

        for i in range(len(object_ids)):
            cur_view = cam_views_t[i]
            # only connected with objects in different view
            possible_match_id = torch.where(cam_views_t != cur_view)[0]

            for j in possible_match_id:
                # TOTO find a way to access world coordinate of the object (there seem to be no corespondance between objects_ids and data.x)
                if torch.dist(data_x_t[i,-3:], data_x_t[j,-3:]) < view_edge_radius:
                    adj_mat[i,j] = True

        adj_mat = adj_mat.to_sparse()._indices()
        adj_mat = og_indices[adj_mat]

    else:
        # logger.info("no view edge filtering")
        adj_mat = torch.ones((len(object_ids),len(object_ids))).bool()

        start = 0
        change_points = np.where(np.diff(cam_views_t))[0] + 1
        for end in change_points:
            adj_mat[start:end, start:end] = False
            start = end
        adj_mat[start:, start:] = False
        adj_mat = adj_mat.to_sparse()._indices()
        adj_mat = og_indices[adj_mat]

    # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()

    #view_edge_list.append(adj_mat)
    return adj_mat, adj_mat_gt


def create_social_edges(data:Data, timepoint=None) -> None:
    #set_trace()
    if "labels_view" not in data.keys():
        raise ValueError("data1 does not contain labels_view")
    node_timestamps = data.labels_view[:,0]
    unique_timestamps = torch.unique(node_timestamps)

    if len(unique_timestamps) == 0:
        data.edge_index_v = torch.tensor([[]], dtype=torch.long)
        data.y_e_v_gt = torch.tensor([[]], dtype=torch.long)
        return

    social_edge_list = []
    seq_index_s_list = []

    for t in unique_timestamps:
        social_edges, seq_indes_s = get_social_edges_for_timepoint(data, t)
        if social_edges is None:
            continue
        social_edge_list.append(social_edges)
        seq_index_s_list.append(seq_indes_s)
    data.edge_index_s = torch.cat(social_edge_list, dim=1)
    data.seq_indices_edge_s = torch.cat(seq_index_s_list, dim=0)

    if timepoint is not None:
        _, seq_indices_s_t = get_social_edges_for_timepoint(data, timepoint)
        data.seq_indices_edge_attr_s = seq_indices_s_t
        data.edge_attr_s = torch.zeros((seq_indices_s_t.shape[0],data.edge_attr.shape[1]), dtype=torch.long).cuda()
    
    
def get_social_edges_for_timepoint(data:Data, timepoint:int) -> torch.Tensor:
    node_timestamps = data.labels_view[:,0]
    xmask = node_timestamps == timepoint
    labels_views_t = data.labels_view[xmask.to(data.labels_view.device)]
    cam_views_t = data.seq_indices_node[xmask.to(data.seq_indices_node.device)]

    object_ids = labels_views_t[:,1]
    if len(object_ids) == 1:
        return None,None
    
    og_indices = xmask.nonzero().reshape(-1) #original indices

    adj_mat = torch.zeros((len(object_ids),len(object_ids)))
    start = 0
    change_points = np.where(np.diff(cam_views_t))[0] + 1
    for end in change_points:
        adj_mat[start:end, start:end] = True
        start = end
    adj_mat[start:, start:] = True
    adj_mat[torch.eye(len(object_ids)).bool()] = False
    adj_mat = adj_mat.to_sparse()._indices()

    #seq_index_s_list.append(cam_views_t[adj_mat[0]])
    seq_indes_s = cam_views_t[adj_mat[0]]

    adj_mat = og_indices[adj_mat]

    #social_edge_list.append(adj_mat)
    social_edges = adj_mat
    return social_edges, seq_indes_s

def get_cluster_map(pyg_data:Data, method="max") -> Tuple[List[Set], Dict]:
    #set_trace()
    G = nx.Graph()
    G.add_edges_from(pyg_data.y_e_v_gt.T.cpu().numpy())
    if pyg_data.y_e_v_gt.shape[1] == 0:
        return torch.tensor([]), {}
    ccs = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    connect_map = {s:d.item() for s, d in enumerate(pyg_data.y_pred_idx.cpu().long())}
    cluster_map = {node.item():i for i, cluster in enumerate(ccs) for node in cluster}
    
    count_dict = {i:{} for i in range(len(ccs))}
    for i, cluster in enumerate(ccs):
        for node in cluster:
            try:
                dst_node = connect_map[node]
            except:
                set_trace()
            if dst_node==-1:
                continue
            if not dst_node in cluster_map:
                set_trace()
            
            dst_cluster = cluster_map[dst_node]
            count_dict[i][dst_cluster] = count_dict[i].get(dst_cluster,0) + 1

    if method=="max":
        c2c = {c: max(cdict,key=cdict.get) for c,cdict in count_dict.items() if len(cdict)>0}
    
    clustermap = torch.tensor([[k,v] for k,v in cluster_map.items()])
    clustermap = clustermap[clustermap[:,0].argsort(dim=0)]
    
    return clustermap, c2c

def get_labels_view_bool(data:Data) -> torch.Tensor:
    labels_views_all = data.edge_index_v.unsqueeze(2)
    labels_views_gt = data.y_e_v_gt.unsqueeze(1)
    return (labels_views_all == labels_views_gt).all(dim=0).any(dim=1).float()
    #(pyg_data.edge_index_v[:,labels_views.bool()]==pyg_data.y_e_v_gt).all()

def id_reass(data:Data):
    #set_trace()
    id =0
    stid2cluster = {}
    cluster2stid = {}

    stid2trajid = {}
    cluster2trajid = {}

    clustids = data.ccs_out[:,0]*10000 + data.ccs_out[:,1]

    final_trajids = -1*torch.ones(data.y_out[:,0].shape[0])
    for i in range(data.y_out.shape[0]):
        stid = data.y_out[i,1].item()
        clustid = clustids[i].item()

        if not stid in stid2cluster:
            stid2cluster[stid] = set()
        stid2cluster[stid].add(clustid)

        if not clustid in cluster2stid:
            cluster2stid[clustid] = set()
        cluster2stid[clustid].add(stid)

        if not stid in stid2trajid and not clustid in cluster2trajid:
            id += 1
            cluster2trajid[clustid] = id
            stid2trajid[stid] = id
        elif not stid in stid2trajid:
            stid2trajid[stid] = cluster2trajid[clustid]
        elif not clustid in cluster2trajid:
            cluster2trajid[clustid] = stid2trajid[stid]
        elif cluster2trajid[clustid] != stid2trajid[stid]:
            continue
            set_trace()
        
        final_trajids[i] = stid2trajid[stid]

    return final_trajids
