import torch
import torch.nn as nn
import torchvision

from model.layers import SimpleMPNN, SimpleEdgeUpdate, SimpleCameraMPNN
from model.utils import get_mlp
from misc.log_utils import log

from graph.graph import HeteroGraph

# class HGnnMPNN(nn.Module):
#     def __init__(self, model_conf, data_conf):
#         super(HGnnMPNN, self).__init__()
#         self.model_conf = model_conf
#         self.data_conf = data_conf

#         self.edge_types = model_conf["edge_types"]
#         self.features = data_conf["features"]
#         self.edge_features = model_conf["edge_features"]
#         self.nhidden = model_conf["dim_hidden_features"]
#         self.num_att_heads = model_conf["num_att_heads"]

#         self.nb_iter_mpnn = model_conf["nb_iter_mpnn"]

#         self.node_feature_mlps = None
#         self.edge_feature_mlps = nn.ModuleList([
#                 nn.ModuleDict(),
#                 nn.ModuleDict()
#             ])

#         self.edge_update_mlps = nn.ModuleDict({
#             edge_type: SimpleEdgeUpdate(self.nhidden, residual=True)
#             for edge_type in self.edge_types
#         })

#         self.mpnns = SimpleMPNN(self.nhidden, self.nhidden, residual=True, bidirectional=False)

#         self.__init_output_layers()


#     def forward(self, graph):

#         hetero_data = graph.data

#         # Perform multiple iterations of message passing
#         for _ in range(self.nb_iter_mpnn):
#             # Perform message passing for each chunk using current attributes
#             updated_node_chunk = self.mpnns(
#                 node_attr=hetero_data['detection'].x,
#                 edge_attr_t=hetero_data['detection', 'temporal', 'detection'].edge_attr,
#                 edge_attr_s=hetero_data['detection', 'social', 'detection'].edge_attr,
#                 edge_attr_v=hetero_data['detection', 'view', 'detection'].edge_attr,
#                 edge_index_t=hetero_data['detection', 'temporal', 'detection'].edge_index,
#                 edge_index_s=hetero_data['detection', 'social', 'detection'].edge_index,
#                 edge_index_v=hetero_data['detection', 'view', 'detection'].edge_index,
#                 num_nodes=hetero_data['detection'].x.size(0)
#             )

#             # Update edge attributes using updated node features
#             for edge_type in self.edge_types:
#                 src, dst = hetero_data[('detection', edge_type, 'detection')].edge_index
#                 src_attr = hetero_data['detection'].x[src]
#                 dst_attr = hetero_data['detection'].x[dst]
#                 edge_attr = hetero_data[('detection', edge_type, 'detection')].edge_attr

#                 updated_edge_attr = self.edge_update_mlps[edge_type](src_attr, dst_attr, edge_attr)
#                 hetero_data[('detection', edge_type, 'detection')].edge_attr = updated_edge_attr

#             # Update node features
#             hetero_data['detection'].x = updated_node_chunk

#         # Prepare output
#         hetero_data['detection'].pred = self.output_transform_node(hetero_data['detection'].x)

#         src_t, dst_t = hetero_data['detection', 'temporal', 'detection'].edge_index
#         src_v, dst_v = hetero_data['detection', 'view', 'detection'].edge_index
        
        
#         hetero_data['detection', 'temporal', 'detection'].edge_pred = self.output_transform_edge(torch.cat([hetero_data['detection'].x[src_t], hetero_data['detection'].x[dst_t], hetero_data['detection', 'temporal', 'detection'].edge_attr], dim=1))
#         # hetero_data['detection', 'social', 'detection'].edge_pred = self.output_transform_edge(hetero_data['detection', 'social', 'detection'].edge_attr)
#         hetero_data['detection', 'view', 'detection'].edge_pred = self.output_transform_edge_v(torch.cat([hetero_data['detection'].x[src_v], hetero_data['detection'].x[dst_v], hetero_data['detection', 'view', 'detection'].edge_attr], dim=1))

#         return {"graph": graph}


#     def __init_output_layers(self):
#         # Output layer for nodes
#         self.output_transform_node = nn.Linear(
#             self.nhidden, 1
#         )
#         # torch.nn.init.kaiming_uniform_(self.output_transform_node.weight, nonlinearity="sigmoid")
#         # self.output_transform_node.bias.data.fill_(4.595)

#         # Output layer for temporal edges
#         self.output_transform_edge = nn.Linear(
#             3*self.nhidden, 1
#         )
#         # torch.nn.init.kaiming_uniform_(self.output_transform_edge.weight, nonlinearity="sigmoid")
#         # self.output_transform_edge.bias.data.fill_(-4.595)

#         # Output layer for view edges
#         self.output_transform_edge_v = nn.Linear(
#             3*self.nhidden, 1
#         )
#         # torch.nn.init.kaiming_uniform_(self.output_transform_edge_v.weight, nonlinearity="sigmoid")
#         # self.output_transform_edge_v.bias.data.fill_(4.595)


#     def output_activation(self, x):
#         return torch.sigmoid(x)

#     def create_initialization_mlps(self, feature_mapping, feature_dict):
#         initialization_mlps = nn.ModuleDict()
#         for feature_name, (start_idx, end_idx) in feature_mapping.items():
#             feature_dim = end_idx - start_idx
#             mlp = get_mlp(
#                 input_dim=feature_dim,
#                 output_dim=feature_dict[feature_name]["hidden_size"],
#                 nhidden=feature_dict[feature_name]["hidden_size"],
#                 nlayer=2,
#                 activation=nn.ReLU,
#                 use_batch_norm=False,
#                 dropout_rate=0.0
#             )
#             # Move the MLP to the same device as the model
#             log.debug(f"MLP device {feature_name}: {next(self.parameters()).device}")
#             mlp = mlp.to(next(self.parameters()).device)
#             initialization_mlps[feature_name] = mlp
#         return initialization_mlps

#     def initialize_node_features(self, features, feature_mapping, agg_type="identity"):
#         if self.node_feature_mlps is None:
#             log.debug(f"Initializing node feature MLPs")
#             self.node_feature_mlps = nn.ModuleList([self.create_initialization_mlps(feature_mapping, self.features), 
#                                                     get_mlp(sum([self.features[feature_name]["hidden_size"] for feature_name in feature_mapping.keys()]), self.nhidden)])

#             log.debug(self.node_feature_mlps)

#         processed_features = []
#         for feature_name, (start_idx, end_idx) in feature_mapping.items():
#             mlp = self.node_feature_mlps[0][feature_name]
#             processed_feature = mlp(features[:, start_idx:end_idx])
#             processed_features.append(processed_feature)

#         concatenated_features = torch.cat(processed_features, dim=1)
#         concatenated_features = self.node_feature_mlps[1](concatenated_features)
    
#         return concatenated_features

#     def initialize_edge_features(self, edge_features, feature_mapping, edge_type, agg_type="mlp"):        
#         if edge_type not in self.edge_feature_mlps[1]:
#             log.debug(f"Initializing edge feature MLPs for edge type: {edge_type}")
#             for feature_name, (start_idx, end_idx) in feature_mapping.items():
#                 if feature_name not in self.edge_feature_mlps[0]:
#                     # Create shared initialization MLPs for all edge types
#                     self.edge_feature_mlps[0][feature_name] = get_mlp(end_idx - start_idx, self.edge_features[edge_type][feature_name]["hidden_size"])
            
#             #Create edge specific MLPs to aggregate features
#             if agg_type == "mlp":
#                 self.edge_feature_mlps[1][edge_type] = get_mlp(sum([self.edge_features[edge_type][feature_name]["hidden_size"] for feature_name in feature_mapping.keys()]), self.nhidden)
#             elif agg_type == "identity":
#                 self.edge_feature_mlps[1][edge_type] = nn.Identity()

#             log.debug(self.edge_feature_mlps)
#         processed_features = []
#         for feature_name, (start_idx, end_idx) in feature_mapping.items():
#             mlp = self.edge_feature_mlps[0][feature_name]
#             processed_feature = mlp(edge_features[:, start_idx:end_idx])
#             processed_features.append(processed_feature)

#         concatenated_features = torch.cat(processed_features, dim=1)
#         edge_features = self.edge_feature_mlps[1][edge_type](concatenated_features)

#         return edge_features

#     def dynamic_size_initialization(self, dataloader):
#         for mv_data in dataloader:
#             temp_graph = HeteroGraph(model_conf=self.model_conf, data_conf=self.data_conf)

#             temp_graph.update_graph(mv_data, self)

#             break

#         log.debug(f"Dynamic size initialization complete")
class HGnnMPNN(nn.Module):
    def __init__(self, model_conf, data_conf):
        super(HGnnMPNN, self).__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.edge_types = model_conf["edge_types"]
        self.features = data_conf["features"]
        self.edge_features = model_conf["edge_features"]
        self.nhidden = model_conf["dim_hidden_features"]
        self.num_att_heads = model_conf["num_att_heads"]

        self.dropout_rate = model_conf["dropout_rate"]

        self.nb_iter_mpnn = model_conf["nb_iter_mpnn"]

        self.init_node_agg_type = model_conf["init_node_agg_type"]
        self.init_edge_agg_type = model_conf["init_edge_agg_type"]

        self.residual_edge_update = model_conf["residual_edge_update"]
        self.residual_node_update = model_conf["residual_node_update"]

        self.nb_layers_mlp_node = model_conf["nb_layers_mlp_node"]
        self.nb_layers_mlp_edge = model_conf["nb_layers_mlp_edge"]  

        self.node_feature_mlps = None
        self.edge_feature_mlps = nn.ModuleList([
                nn.ModuleDict(),
                nn.ModuleDict()
            ])

        self.crop_feature_extractor_type = model_conf["crop_feature_extractor_type"]
        self.crops_feature_extractor = None

        self.edge_update_mlps = nn.ModuleDict({
            edge_type: SimpleEdgeUpdate(self.nhidden, self.nhidden, nb_mlp_layers=self.nb_layers_mlp_edge, residual=self.residual_edge_update, dropout_rate=self.dropout_rate)
            for edge_type in self.edge_types
        })

        self.mpnns = SimpleMPNN(self.nhidden, self.nhidden, nb_mlp_layers=self.nb_layers_mlp_node, residual=self.residual_node_update, dropout_rate=self.dropout_rate)

        self.__init_output_layers()


    def forward(self, graph):

        hetero_data = graph.data

        # Perform multiple iterations of message passing
        for _ in range(self.nb_iter_mpnn):
            # Perform message passing for each chunk using current attributes
            updated_node_chunk = self.mpnns(
                node_attr=hetero_data['detection'].x,
                edge_attr_t=hetero_data['detection', 'temporal', 'detection'].edge_attr,
                edge_attr_s=hetero_data['detection', 'social', 'detection'].edge_attr,
                edge_attr_v=hetero_data['detection', 'view', 'detection'].edge_attr,
                edge_index_t=hetero_data['detection', 'temporal', 'detection'].edge_index,
                edge_index_s=hetero_data['detection', 'social', 'detection'].edge_index,
                edge_index_v=hetero_data['detection', 'view', 'detection'].edge_index,
                num_nodes=hetero_data['detection'].x.size(0)
            )

            # Update edge attributes using updated node features
            for edge_type in self.edge_types:
                src, dst = hetero_data[('detection', edge_type, 'detection')].edge_index
                src_attr = hetero_data['detection'].x[src]
                dst_attr = hetero_data['detection'].x[dst]
                edge_attr = hetero_data[('detection', edge_type, 'detection')].edge_attr

                updated_edge_attr = self.edge_update_mlps[edge_type](src_attr, dst_attr, edge_attr)
                hetero_data[('detection', edge_type, 'detection')].edge_attr = updated_edge_attr

            # Update node features
            hetero_data['detection'].x = updated_node_chunk

        # Prepare output
        hetero_data['detection'].pred = self.output_transform_node(hetero_data['detection'].x)
        # hetero_data['detection', 'temporal', 'detection'].edge_pred = self.output_transform_edge(hetero_data['detection', 'temporal', 'detection'].edge_attr)
        # # hetero_data['detection', 'social', 'detection'].edge_pred = self.output_transform_edge(hetero_data['detection', 'social', 'detection'].edge_attr)
        # hetero_data['detection', 'view', 'detection'].edge_pred = self.output_transform_edge_v(hetero_data['detection', 'view', 'detection'].edge_attr)

        src_t, dst_t = hetero_data['detection', 'temporal', 'detection'].edge_index
        src_v, dst_v = hetero_data['detection', 'view', 'detection'].edge_index
        hetero_data['detection', 'temporal', 'detection'].edge_pred = self.output_transform_edge(torch.cat([hetero_data['detection'].x[src_t], hetero_data['detection'].x[dst_t], hetero_data['detection', 'temporal', 'detection'].edge_attr], dim=1))
        # hetero_data['detection', 'social', 'detection'].edge_pred = self.output_transform_edge(hetero_data['detection', 'social', 'detection'].edge_attr)
        hetero_data['detection', 'view', 'detection'].edge_pred = self.output_transform_edge_v(torch.cat([hetero_data['detection'].x[src_v], hetero_data['detection'].x[dst_v], hetero_data['detection', 'view', 'detection'].edge_attr], dim=1))
        
        return {"graph": graph}


    # def __init_output_layers(self):
    #     # Output layer for nodes
    #     self.output_transform_node = nn.Linear(
    #         self.nhidden, 1
    #     )
    #     # torch.nn.init.kaiming_uniform_(self.output_transform_node.weight, nonlinearity="sigmoid")
    #     # self.output_transform_node.bias.data.fill_(4.595)

    #     # Output layer for temporal edges
    #     self.output_transform_edge = nn.Linear(
    #         self.nhidden, 1
    #     )
    #     # torch.nn.init.kaiming_uniform_(self.output_transform_edge.weight, nonlinearity="sigmoid")
    #     # self.output_transform_edge.bias.data.fill_(-4.595)

    #     # Output layer for view edges
    #     self.output_transform_edge_v = nn.Linear(
    #         self.nhidden, 1
    #     )
    #     torch.nn.init.kaiming_uniform_(self.output_transform_edge_v.weight, nonlinearity="sigmoid")
    #     self.output_transform_edge_v.bias.data.fill_(4.595)

    def __init_output_layers(self):
        # Output layer for nodes
        self.output_transform_node = nn.Linear(
            self.nhidden, 1
        )
        # torch.nn.init.kaiming_uniform_(self.output_transform_node.weight, nonlinearity="sigmoid")
        # self.output_transform_node.bias.data.fill_(4.595)

        # Output layer for temporal edges
        self.output_transform_edge = nn.Linear(
            3*self.nhidden, 1
        )
        # torch.nn.init.kaiming_uniform_(self.output_transform_edge.weight, nonlinearity="sigmoid")
        # self.output_transform_edge.bias.data.fill_(-4.595)

        # Output layer for view edges
        self.output_transform_edge_v = nn.Linear(
            3*self.nhidden, 1
        )
        # torch.nn.init.kaiming_uniform_(self.output_transform_edge_v.weight, nonlinearity="sigmoid")
        # self.output_transform_edge_v.bias.data.fill_(4.595)


    def output_activation(self, x):
        return torch.sigmoid(x)

    def create_initialization_mlps(self, feature_mapping, feature_dict):
        initialization_mlps = nn.ModuleDict()
        for feature_name, (start_idx, end_idx) in feature_mapping.items():
            feature_dim = end_idx - start_idx
            mlp = get_mlp(
                input_dim=feature_dim,
                output_dim=feature_dict[feature_name]["hidden_size"],
                nhidden=feature_dict[feature_name]["hidden_size"],
                nlayer=2,
                activation=nn.ReLU,
                use_batch_norm=False,
                dropout_rate=0.0
            )
            # Move the MLP to the same device as the model
            log.debug(f"MLP device {feature_name}: {next(self.parameters()).device}")
            mlp = mlp.to(next(self.parameters()).device)
            initialization_mlps[feature_name] = mlp
        return initialization_mlps

    def initialize_node_features(self, features, feature_mapping):
        if self.node_feature_mlps is None:
            log.debug(f"Initializing node feature MLPs")
            self.node_feature_mlps = nn.ModuleList([self.create_initialization_mlps(feature_mapping, self.features)])
            
            if self.init_node_agg_type == "mlp":
                self.node_feature_mlps.append(get_mlp(sum([self.features[feature_name]["hidden_size"] for feature_name in feature_mapping.keys()]), self.nhidden))
            elif self.init_node_agg_type == "identity":
                self.node_feature_mlps.append(nn.Identity())
            
            log.debug(self.node_feature_mlps)

        processed_features = []
        for feature_name, (start_idx, end_idx) in feature_mapping.items():
            mlp = self.node_feature_mlps[0][feature_name]
            processed_feature = mlp(features[:, start_idx:end_idx])
            processed_features.append(processed_feature)

        concatenated_features = torch.cat(processed_features, dim=1)
        
        if self.init_node_agg_type == "mlp":
            concatenated_features = self.node_feature_mlps[1](concatenated_features)
        elif self.init_node_agg_type == "identity":
            # If identity, we don't need to do anything further
            pass
    
        return concatenated_features

    def initialize_crops(self, crops):
        if self.crops_feature_extractor is None:
            if self.crop_feature_extractor_type == "resnet18":
                resnet = torchvision.models.resnet18(weights='DEFAULT')
                self.crops_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            elif self.crop_feature_extractor_type == "simple_cnn":
                self.crops_feature_extractor = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
                )
            elif self.crop_feature_extractor_type == "resnet34":
                resnet = torchvision.models.resnet34(weights='DEFAULT')
                self.crops_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            elif self.crop_feature_extractor_type == "resnet50":
                resnet = torchvision.models.resnet50(weights='DEFAULT')
                self.crops_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            elif self.crop_feature_extractor_type == "resnet101":
                resnet = torchvision.models.resnet101(weights='DEFAULT')
                self.crops_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        crops = self.crops_feature_extractor(crops).view(crops.size(0), -1)
            
        return crops

    def initialize_edge_features(self, edge_features, feature_mapping, edge_type, agg_type="mlp"):        
        if edge_type not in self.edge_feature_mlps[1]:
            log.debug(f"Initializing edge feature MLPs for edge type: {edge_type}")
            for feature_name, (start_idx, end_idx) in feature_mapping.items():
                if feature_name not in self.edge_feature_mlps[0]:
                    # Create shared initialization MLPs for all edge types
                    self.edge_feature_mlps[0][feature_name] = get_mlp(end_idx - start_idx, self.edge_features[edge_type][feature_name]["hidden_size"])
            
            #Create edge specific MLPs to aggregate features
            if self.init_edge_agg_type == "mlp":
                self.edge_feature_mlps[1][edge_type] = get_mlp(sum([self.edge_features[edge_type][feature_name]["hidden_size"] for feature_name in feature_mapping.keys()]), self.nhidden)
            elif self.init_edge_agg_type == "identity":
                self.edge_feature_mlps[1][edge_type] = nn.Identity()

            log.debug(self.edge_feature_mlps)
        processed_features = []
        for feature_name, (start_idx, end_idx) in feature_mapping.items():
            mlp = self.edge_feature_mlps[0][feature_name]
            processed_feature = mlp(edge_features[:, start_idx:end_idx])
            processed_features.append(processed_feature)

        concatenated_features = torch.cat(processed_features, dim=1)
        if self.init_edge_agg_type == "mlp":
            edge_features = self.edge_feature_mlps[1][edge_type](concatenated_features)
        elif self.init_edge_agg_type == "identity":
            edge_features = concatenated_features

        return edge_features

    def dynamic_size_initialization(self, dataloader):
        for mv_data in dataloader:
            temp_graph = HeteroGraph(model_conf=self.model_conf, data_conf=self.data_conf)

            temp_graph.update_graph(mv_data, self)

            break

        log.debug(f"Dynamic size initialization complete")



class HGnnCameraMPNN(nn.Module):
    def __init__(self, model_conf, data_conf):
        super(HGnnCameraMPNN, self).__init__()
        self.model_conf = model_conf
        self.data_conf = data_conf

        self.edge_types = model_conf["edge_types"] 
        self.features = data_conf["features"]
        self.edge_features = model_conf["edge_features"]
        self.nhidden = model_conf["dim_hidden_features"]
        self.num_att_heads = model_conf["num_att_heads"]

        self.dropout_rate = model_conf["dropout_rate"]

        self.nb_iter_mpnn = model_conf["nb_iter_mpnn"]

        self.init_node_agg_type = model_conf["init_node_agg_type"]
        self.init_edge_agg_type = model_conf["init_edge_agg_type"]
        self.init_camera_agg_type = model_conf["init_camera_agg_type"]
        
        self.residual_edge_update = model_conf["residual_edge_update"]
        self.residual_node_update = model_conf["residual_node_update"]

        self.nb_layers_mlp_node = model_conf["nb_layers_mlp_node"]
        self.nb_layers_mlp_edge = model_conf["nb_layers_mlp_edge"]

        self.node_feature_mlps = None
        self.edge_feature_mlps = nn.ModuleList([
                nn.ModuleDict(),
                nn.ModuleDict()
            ])

        self.crops_feature_extractor = None

        self.camera_edge_features = model_conf["camera_edge_features"]
        self.camera_feature_mlps = nn.ModuleDict()
        self.camera_edge_feature_mlps = nn.ModuleDict()

        self.edge_update_mlps = nn.ModuleDict({
            edge_type: SimpleEdgeUpdate(self.nhidden, self.nhidden, nb_mlp_layers=self.nb_layers_mlp_edge, residual=self.residual_edge_update, dropout_rate=self.dropout_rate)
            for edge_type in self.edge_types + ["see", "next"]
        })

        self.mpnns = SimpleCameraMPNN(self.nhidden, self.nhidden, nb_mlp_layers=self.nb_layers_mlp_node, residual=self.residual_node_update, dropout_rate=self.dropout_rate)

        self.__init_output_layers()


    def forward(self, graph):

        hetero_data = graph.data

        # Perform multiple iterations of message passing
        for _ in range(self.nb_iter_mpnn):
            # Perform message passing for each chunk using current attributes
            updated_node_chunk, updated_camera_node_chunk = self.mpnns(
                node_attr=hetero_data['detection'].x,
                edge_attr_t=hetero_data['detection', 'temporal', 'detection'].edge_attr,
                edge_attr_s=hetero_data['detection', 'social', 'detection'].edge_attr,
                edge_attr_v=hetero_data['detection', 'view', 'detection'].edge_attr,
                camera_node_attr=hetero_data['camera'].x,
                camera_see_edge_attr=hetero_data['camera', 'see', 'detection'].edge_attr if hasattr(hetero_data['camera', 'see', 'detection'], 'edge_attr') else None,
                camera_next_edge_attr=hetero_data['camera', 'next', 'camera'].edge_attr if hasattr(hetero_data['camera', 'next', 'camera'], 'edge_attr') else None,
                edge_index_t=hetero_data['detection', 'temporal', 'detection'].edge_index,
                edge_index_s=hetero_data['detection', 'social', 'detection'].edge_index,
                edge_index_v=hetero_data['detection', 'view', 'detection'].edge_index,
                edge_index_see=hetero_data['camera', 'see', 'detection'].edge_index if hasattr(hetero_data['camera', 'see', 'detection'], 'edge_index') else None,
                edge_index_next=hetero_data['camera', 'next', 'camera'].edge_index if hasattr(hetero_data['camera', 'next', 'camera'], 'edge_index') else None,
                num_nodes=hetero_data['detection'].x.size(0),
                num_cameras=hetero_data['camera'].x.size(0)
            )
       
            # Update edge attributes using updated node features
            for edge_type in self.edge_types:
                src, dst = hetero_data[('detection', edge_type, 'detection')].edge_index
                src_attr = hetero_data['detection'].x[src]
                dst_attr = hetero_data['detection'].x[dst]
                edge_attr = hetero_data[('detection', edge_type, 'detection')].edge_attr

                updated_edge_attr = self.edge_update_mlps[edge_type](src_attr, dst_attr, edge_attr)
                hetero_data[('detection', edge_type, 'detection')].edge_attr = updated_edge_attr
            # Update camera edge features
            for edge_type in ["see", "next"]:
                if edge_type == "next" and not hasattr(hetero_data['camera', 'next', 'camera'], 'edge_index'):
                    continue
                src, dst = hetero_data[('camera', edge_type, 'camera' if edge_type == 'next' else 'detection')].edge_index
                src_attr = hetero_data['camera'].x[src]
                dst_attr = hetero_data['camera' if edge_type == 'next' else 'detection'].x[dst]
                edge_attr = hetero_data[('camera', edge_type, 'camera' if edge_type == 'next' else 'detection')].edge_attr

                updated_edge_attr = self.edge_update_mlps[edge_type](src_attr, dst_attr, edge_attr)
                hetero_data[('camera', edge_type, 'camera' if edge_type == 'next' else 'detection')].edge_attr = updated_edge_attr

            # Update node features
            hetero_data['detection'].x = updated_node_chunk
            hetero_data['camera'].x = updated_camera_node_chunk
        # Prepare output
        hetero_data['detection'].pred = self.output_transform_node(hetero_data['detection'].x)
        # hetero_data['detection', 'temporal', 'detection'].edge_pred = self.output_transform_edge(hetero_data['detection', 'temporal', 'detection'].edge_attr)
        # # hetero_data['detection', 'social', 'detection'].edge_pred = self.output_transform_edge(hetero_data['detection', 'social', 'detection'].edge_attr)
        # hetero_data['detection', 'view', 'detection'].edge_pred = self.output_transform_edge_v(hetero_data['detection', 'view', 'detection'].edge_attr)

        src_t, dst_t = hetero_data['detection', 'temporal', 'detection'].edge_index
        src_v, dst_v = hetero_data['detection', 'view', 'detection'].edge_index
        hetero_data['detection', 'temporal', 'detection'].edge_pred = self.output_transform_edge(torch.cat([hetero_data['detection'].x[src_t], hetero_data['detection'].x[dst_t], hetero_data['detection', 'temporal', 'detection'].edge_attr], dim=1))
        # hetero_data['detection', 'social', 'detection'].edge_pred = self.output_transform_edge(hetero_data['detection', 'social', 'detection'].edge_attr)
        hetero_data['detection', 'view', 'detection'].edge_pred = self.output_transform_edge_v(torch.cat([hetero_data['detection'].x[src_v], hetero_data['detection'].x[dst_v], hetero_data['detection', 'view', 'detection'].edge_attr], dim=1))
        
        return {"graph": graph}


    # def __init_output_layers(self):
    #     # Output layer for nodes
    #     self.output_transform_node = nn.Linear(
    #         self.nhidden, 1
    #     )
    #     # torch.nn.init.kaiming_uniform_(self.output_transform_node.weight, nonlinearity="sigmoid")
    #     # self.output_transform_node.bias.data.fill_(4.595)

    #     # Output layer for temporal edges
    #     self.output_transform_edge = nn.Linear(
    #         self.nhidden, 1
    #     )
    #     # torch.nn.init.kaiming_uniform_(self.output_transform_edge.weight, nonlinearity="sigmoid")
    #     # self.output_transform_edge.bias.data.fill_(-4.595)

    #     # Output layer for view edges
    #     self.output_transform_edge_v = nn.Linear(
    #         self.nhidden, 1
    #     )
    #     torch.nn.init.kaiming_uniform_(self.output_transform_edge_v.weight, nonlinearity="sigmoid")
    #     self.output_transform_edge_v.bias.data.fill_(4.595)

    def __init_output_layers(self):
        # Output layer for nodes
        self.output_transform_node = nn.Linear(
            self.nhidden, 1
        )
        # torch.nn.init.kaiming_uniform_(self.output_transform_node.weight, nonlinearity="sigmoid")
        # self.output_transform_node.bias.data.fill_(4.595)

        # Output layer for temporal edges
        self.output_transform_edge = nn.Linear(
            3*self.nhidden, 1
        )
        # torch.nn.init.kaiming_uniform_(self.output_transform_edge.weight, nonlinearity="sigmoid")
        # self.output_transform_edge.bias.data.fill_(-4.595)

        # Output layer for view edges
        self.output_transform_edge_v = nn.Linear(
            3*self.nhidden, 1
        )
        # torch.nn.init.kaiming_uniform_(self.output_transform_edge_v.weight, nonlinearity="sigmoid")
        # self.output_transform_edge_v.bias.data.fill_(4.595)


    def output_activation(self, x):
        return torch.sigmoid(x)

    def create_initialization_mlps(self, feature_mapping, feature_dict):
        initialization_mlps = nn.ModuleDict()
        for feature_name, (start_idx, end_idx) in feature_mapping.items():
            feature_dim = end_idx - start_idx
            mlp = get_mlp(
                input_dim=feature_dim,
                output_dim=feature_dict[feature_name]["hidden_size"],
                nhidden=feature_dict[feature_name]["hidden_size"],
                nlayer=2,
                activation=nn.ReLU,
                use_batch_norm=False,
                dropout_rate=0.0
            )
            # Move the MLP to the same device as the model
            log.debug(f"MLP device {feature_name}: {next(self.parameters()).device}")
            mlp = mlp.to(next(self.parameters()).device)
            initialization_mlps[feature_name] = mlp
        return initialization_mlps

    def initialize_node_features(self, features, feature_mapping):
        if self.node_feature_mlps is None:
            log.debug(f"Initializing node feature MLPs")
            self.node_feature_mlps = nn.ModuleList([self.create_initialization_mlps(feature_mapping, self.features)])
            
            if self.init_node_agg_type == "mlp":
                self.node_feature_mlps.append(get_mlp(sum([self.features[feature_name]["hidden_size"] for feature_name in feature_mapping.keys()]), self.nhidden))
            elif self.init_node_agg_type == "identity":
                self.node_feature_mlps.append(nn.Identity())
            
            log.debug(self.node_feature_mlps)

        processed_features = []
        for feature_name, (start_idx, end_idx) in feature_mapping.items():
            mlp = self.node_feature_mlps[0][feature_name]
            processed_feature = mlp(features[:, start_idx:end_idx])
            processed_features.append(processed_feature)

        concatenated_features = torch.cat(processed_features, dim=1)
        
        if self.init_node_agg_type == "mlp":
            concatenated_features = self.node_feature_mlps[1](concatenated_features)
        elif self.init_node_agg_type == "identity":
            # If identity, we don't need to do anything further
            pass
    
        return concatenated_features

    def initialize_crops(self, crops):
        if self.crops_feature_extractor is None:
            # resnet = torchvision.models.resnet18(weights='DEFAULT')
            # self.crops_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            self.crops_feature_extractor = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )

        crops = self.crops_feature_extractor(crops).view(crops.size(0), -1)
            
        return crops

    def initialize_edge_features(self, edge_features, feature_mapping, edge_type, agg_type="mlp"):        
        if edge_type not in self.edge_feature_mlps[1]:
            log.debug(f"Initializing edge feature MLPs for edge type: {edge_type}")
            for feature_name, (start_idx, end_idx) in feature_mapping.items():
                if feature_name not in self.edge_feature_mlps[0]:
                    # Create shared initialization MLPs for all edge types
                    self.edge_feature_mlps[0][feature_name] = get_mlp(end_idx - start_idx, self.edge_features[edge_type][feature_name]["hidden_size"])
            
            #Create edge specific MLPs to aggregate features
            if self.init_edge_agg_type == "mlp":
                self.edge_feature_mlps[1][edge_type] = get_mlp(sum([self.edge_features[edge_type][feature_name]["hidden_size"] for feature_name in feature_mapping.keys()]), self.nhidden)
            elif self.init_edge_agg_type == "identity":
                self.edge_feature_mlps[1][edge_type] = nn.Identity()

            log.debug(self.edge_feature_mlps)
        processed_features = []
        for feature_name, (start_idx, end_idx) in feature_mapping.items():
            mlp = self.edge_feature_mlps[0][feature_name]
            processed_feature = mlp(edge_features[:, start_idx:end_idx])
            processed_features.append(processed_feature)

        concatenated_features = torch.cat(processed_features, dim=1)
        if self.init_edge_agg_type == "mlp":
            edge_features = self.edge_feature_mlps[1][edge_type](concatenated_features)
        elif self.init_edge_agg_type == "identity":
            edge_features = concatenated_features

        return edge_features

    def initialize_camera_features(self, features, feature_mapping, feature_dict):
        processed_features = []
        for feature_name, (start_idx, end_idx) in feature_mapping.items():
            if feature_name not in self.camera_feature_mlps:
                hidden_size = feature_dict[feature_name].get("hidden_size", self.nhidden)
                self.camera_feature_mlps[feature_name] = get_mlp(end_idx - start_idx, hidden_size)
            
            mlp = self.camera_feature_mlps[feature_name]
            processed_feature = mlp(features[:, start_idx:end_idx])
            processed_features.append(processed_feature)

        concatenated_features = torch.cat(processed_features, dim=1)
        
        if self.init_camera_agg_type == "mlp":
            if "camera_agg" not in self.camera_feature_mlps:
                total_size = sum([feature_dict[f].get("hidden_size", self.nhidden) for f in feature_mapping.keys()])
                self.camera_feature_mlps["camera_agg"] = get_mlp(total_size, self.nhidden)
            concatenated_features = self.camera_feature_mlps["camera_agg"](concatenated_features)
        elif self.init_camera_agg_type == "identity":
            # If identity, we don't need to do anything further
            pass
    
        return concatenated_features

    def initialize_camera_edge_features(self, edge_attr, feature_mapping, edge_type):
        processed_features = []
        for feature_name, (start_idx, end_idx) in feature_mapping.items():
            if feature_name not in self.camera_edge_feature_mlps:
                hidden_size = 1 if edge_type == "next" else self.camera_edge_features[feature_name].get("hidden_size", self.nhidden)
                self.camera_edge_feature_mlps[feature_name] = get_mlp(end_idx - start_idx, hidden_size)

                log.debug(f"Initialized {feature_name} edge feature MLP for camera edges")
            
            mlp = self.camera_edge_feature_mlps[feature_name]
            processed_feature = mlp(edge_attr[:, start_idx:end_idx])
            processed_features.append(processed_feature)

        concatenated_features = torch.cat(processed_features, dim=1)
        
        if self.init_camera_agg_type == "mlp":
            if edge_type == "next":
                if "camera_edge_agg_next" not in self.camera_edge_feature_mlps:
                    total_size = 1
                    self.camera_edge_feature_mlps["camera_edge_agg_next"] = get_mlp(total_size, self.nhidden)
                    log.debug(f"Initialized camera_edge_agg_next edge feature MLP")

                concatenated_features = self.camera_edge_feature_mlps["camera_edge_agg_next"](concatenated_features)
            elif edge_type == "see":
                if "camera_edge_agg_see" not in self.camera_edge_feature_mlps:
                    total_size = sum([self.camera_edge_features[f].get("hidden_size", self.nhidden) for f in feature_mapping.keys()])
                    self.camera_edge_feature_mlps["camera_edge_agg_see"] = get_mlp(total_size, self.nhidden)
                    log.debug(f"Initialized camera_edge_agg_see edge feature MLP")

                concatenated_features = self.camera_edge_feature_mlps["camera_edge_agg_see"](concatenated_features)
        elif self.init_camera_agg_type == "identity":
            # If identity, we don't need to do anything further
            pass
    
        return concatenated_features

    def dynamic_size_initialization(self, dataloader, init_iterations=1):
        for i, mv_data in enumerate(dataloader):
            temp_graph = HeteroGraph(model_conf=self.model_conf, data_conf=self.data_conf)

            temp_graph.update_graph(mv_data, self)

            if i >= init_iterations - 1:
                break

        edge_attr = torch.zeros(2, 1, device=temp_graph.device)
        feature_mapping = {f'zeros': (0, 1)}
        temp_graph.data['camera', 'next', 'camera'].edge_attr = self.initialize_camera_edge_features(edge_attr, feature_mapping, "next")

        log.debug(f"Dynamic size initialization complete")