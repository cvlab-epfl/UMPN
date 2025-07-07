
# GRAPH_NODE_REL_ATTR = ['labels_n', 'y_pred', 'labels_view', 'seq_indices_node', 'h_out_n', 'y_n_act', 'y_n']

# GRAPH_EDGE_REL_ATTR = ['edge_index', 'labels_e', 'seq_indices_edge', 'h_out_e', 'y_e_act', 'y_e']

# SEQ_INDICES_MAPPING = {
#     'seq_indices_node': 'labels_n',
#     'seq_indices_edge': 'labels_e',
#     'seq_indices_x': 'x',
#     'seq_indices_edge_attr': 'edge_attr',
# }


SEQ_INDICES_MAPPING = {
    'seq_indices_node': ['labels_n', 'y_pred', 'labels_view', 'seq_indices_node', 'h_out_n', 'y_n_act', 'y_n', 'y_pred_idx','ccs'],
    'seq_indices_edge': ['edge_index', 'labels_e', 'seq_indices_edge', 'h_out_e', 'y_e_act', 'y_e'],
    #'seq_indices_edge_s': ['h_out_e','seq_indices_edge_s'],
    'seq_indices_x': ['x','seq_indices_x'],
    'seq_indices_edge_attr': ['edge_attr','seq_indices_edge_attr'],
    'seq_indices_edge_attr_s': ['edge_attr_s','seq_indices_edge_attr_s'],

    'seq_indices_y_out': ['y_out','ccs_out','seq_indices_y_out'],
}

PLOT_COLORS = {
    'temporal': {"true": "green", "pred": "violet", "false": "red"},
    'view': {"true": "blue", "pred": "yellow", "false": "red"},
}


CUMULATE_ATTR = "labels_n"
OFFSET_ATTR = "seq_indices_node"