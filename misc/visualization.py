import torch

from torch_geometric.data import Data
import datetime
import os
import ipdb
from tqdm import tqdm
from graph.visualization import get_color_for_instance
import numpy as np
import matplotlib.pyplot as plt
from misc.log_utils import log

def save_matimg(matrix, name=None):
    import matplotlib.pyplot as plt

    N = matrix.shape[0]
    try:
        plt.close()
    except:
        pass
    plt.figure(dpi=300)
    if isinstance(matrix, torch.Tensor):
        if matrix.requires_grad:
            if isinstance(matrix, torch.sparse.FloatTensor) or isinstance(
                matrix, torch.cuda.sparse.FloatTensor
            ):
                plt.imshow(matrix.to_dense().detach().cpu())
            else:
                plt.imshow(matrix.detach().cpu())
        else:
            if isinstance(matrix, torch.sparse.FloatTensor) or isinstance(
                matrix, torch.cuda.sparse.FloatTensor
            ):
                plt.imshow(matrix.to_dense().cpu())
            else:
                plt.imshow(matrix.cpu())
    else:
        plt.imshow(matrix)

    ax = plt.gca()

    # Remove the spines while keeping the ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    path = f"viz/matrices/{datetime.datetime.now().strftime('%m-%d')}/"
    import os

    if not os.path.exists(path):
        os.makedirs(path)
    if name is None:
        plt.savefig(path + f"matrix{datetime.datetime.now()}.jpg")
    else:
        plt.savefig(path + f"{name}{datetime.datetime.now()}.jpg")
    plt.close()




def plot_graph(y_pred, feats, node_adj, edge_adj, labels, t1, tN):
    import matplotlib.pyplot as plt
    import datetime

    assert feats.shape[1] == 9, "Make sure to include spatial features!"

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(30, 20), dpi=80)
    # set_trace()
    feats, node_adj, edge_adj, labels = (
        feats.cpu().numpy(),
        node_adj.numpy(),
        edge_adj.numpy(),
        labels.numpy(),
    )
    # Create a dictionary to store color and last location for each person
    is_edge = y_pred[:, 0] == -1
    is_node = ~is_edge
    for i, feat in enumerate(feats[is_node]):
        x, y = feat[6], feat[7]
        if x > 0 and y > 0:
            alpha_val = 1
            plt.scatter(x, y, alpha=alpha_val, label=f"Detection")
    for i, nodelist in enumerate(node_adj[is_edge]):
        src = nodelist == 1
        dst = nodelist == -1
        feat_src = feats[src][0]
        feat_dst = feats[dst][0]

        edge_feat = (feat_dst + feat_src) / 2
        x_e, y_e = edge_feat[6], edge_feat[7]
        x_s, y_s = feat_src[6], feat_src[7]
        x_d, y_d = feat_dst[6], feat_dst[7]
        if x > 0 and y > 0:
            # plt.scatter(x_e, y_e, alpha=alpha_val, label=f"Edge")
            plt.plot([x_s, x_d], [y_s, y_d], color="red", alpha=0.8)
    plt.xlabel("Xw")
    plt.ylabel("Yw")
    plt.title("Person Positions")
    plt.ylim(0, 2000)
    plt.xlim(0, 1000)
    plt.savefig(f"viz/graphs/init_graph_{datetime.datetime.now()}.png")
    plt.close()



def plot_graph_pyg(graph:Data,seq:int=None,name:str="",ignore_negative:bool=True,edge_type:str="temporal",annotate:bool=True):
    from ipdb import set_trace
    
    #set_trace()
    
    if edge_type not in ["temporal","view"]:
        raise NotImplementedError
    
    import matplotlib.pyplot as plt
    if seq is not None and "seq_indices_node" in graph.keys:
        from utils.multiview_helper import extract_sequence
        data = extract_sequence(graph,seq)
    else:
        data = graph

    assert data.x.shape[1] == 9, "Make sure to include spatial features!"

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(30, 20), dpi=120)
    # set_trace()
    feats = data.x.cpu().numpy()
    if edge_type == "temporal":
        edge_index = data.edge_index.cpu().numpy() 
    elif edge_type == "view":
        edge_index = data.y_e_v_gt.cpu().numpy()
    # Create a dictionary to store color and last location for each person

    for i, feat in enumerate(feats):
        timestamp = data.labels_view[i,0].long().item()
        objid = data.labels_view[i,1].long().item()
        viewid = f"v{data.seq_indices_node[i].item()}" if "seq_indices_node" in data.keys else ""
        obj_name = f"t{timestamp}id{objid}{viewid}"
        x, y = feat[6], feat[7]
        alpha_val = 1
        plt.scatter(x, y, alpha=alpha_val, label=f"Detection",color="orange")
        if annotate:
            plt.annotate(obj_name, (x,y),alpha=0.8,size=10)#,bbox = dict(boxstyle="round", fc="0.1"))
    
    for e_i, e in enumerate(edge_index.T):

        if edge_type == "temporal":
            is_traj = data.labels_e[i].item()
            if ignore_negative and not is_traj:
                continue
            color = "green" if is_traj else "red"
            marker = "-x"
        elif edge_type == "view":
            # is_traj = data.labels_e_v_gt[i].item()
            # if ignore_negative and not is_traj:
            #     continue
            color = "blue"# if is_traj else "red"
            marker = "-"

        #pyg_data.y_pred[:, 1] ----> pyg_data.y_out index == bbox_pred index
        
        src, dst= e
        

        feat_src = feats[src]
        feat_dst = feats[dst]

        edge_feat = (feat_dst + feat_src) / 2
        x_e, y_e = edge_feat[6], edge_feat[7]
        x_s, y_s = feat_src[6], feat_src[7]
        x_d, y_d = feat_dst[6], feat_dst[7]
        if (x_s > 0 and y_s > 0) or (x_d > 0 and y_d > 0):
            # plt.scatter(x_e, y_e, alpha=alpha_val, label=f"Edge")
            plt.plot([x_s, x_d], [y_s, y_d], marker,color=color, alpha=0.6)
        else:
            print(f"skipping edge x_s:{x_s:.2f}, y_s:{y_s:.2f}, x_d:{x_d:.2f}, y_d:{y_d:.2f}")

    seq_str = "" if seq is None else f"_seq{seq}"

    plt.xlabel("Xw")
    plt.ylabel("Yw")
    plt.title("Person Positions")
    plt.ylim(0, 2000)
    plt.xlim(0, 1000)
    plt.savefig(f"viz/graphs/{datetime.datetime.now()}_{name}{seq_str}_{edge_type}.pdf")
    plt.close()


def plot_wildtrack_detections_no_track_pred(bbox_pred, bbox_gt, y_out, y_seq,name:str="",seq_indices=None,seq_indices_gt=None,movie=False,seqid=None):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 12))
    #ipdb.set_trace()
    location_pred = bbox_pred[:, 9:11]  # x,y in world coordinates
    location_gt = bbox_gt[:, 9:11]  # x,y in world coordinates
    # keep_ids = y_out[:,1]!=-1
    # y_out = y_out[keep_ids]
    # y_seq = y_seq[0][keep_ids]
    id_pred = y_out[:, 1]
    id_gt = y_seq[:, 1]
    frame_ids = y_seq[:, 0]  # frame id's

    if y_seq.ndim == 3:
        id_gt = y_seq[0,:, 1]
        frame_ids = y_seq[0,:, 0]  # frame id's

    if y_out.ndim == 3:
        id_pred = y_out[0,:, 1]
    from ipdb import set_trace
    #set_trace()
    if seq_indices is not None:
        rng = torch.unique(seq_indices)        
    if seqid is not None:
        rng = [seqid]

    for i, person_id in enumerate(np.unique(id_gt)):
        for seq in rng:
            if person_id == -1:
                continue
            idxs = (id_gt == person_id) & (frame_ids >= 360)
            if seq_indices_gt is not None:
                idxs = idxs & (seq_indices_gt.squeeze() == seq).numpy()  # indices for the same person id
            idxs = np.where(idxs)
            sorted_idxs = idxs[0][np.argsort(frame_ids[idxs])]  # sort indices by frame_id
            plt.plot(
                location_gt[sorted_idxs, 0],
                location_gt[sorted_idxs, 1],
                "o-",
                label=f"Person {person_id}",
                linewidth=3,
                alpha=0.05,
                color="black",
                markersize=3
            )

    cmap = plt.get_cmap("tab20")

    frame_ids = y_out[:, 0]  # frame id's
    x = len(np.unique(frame_ids))
    # Generate an array of equally spaced values between 0 and 1
    values = np.linspace(0, 1, x)

    # Use a colormap to get a color for each value
    colormap = plt.cm.viridis  # you can change this to other colormaps like plasma, inferno, etc.
    colors = [colormap(val) for val in values]
    name = f"{datetime.datetime.now()}_{name}"
    import os
    if not os.path.exists(f"viz/graphs/{name}/"):
        os.makedirs(f"viz/graphs/{name}/")
    plt.tight_layout()

    savepath = f"viz/graphs/{datetime.datetime.now().strftime('%m-%d')}"

    for i, f_id in enumerate(np.unique(frame_ids)):
        for seq in rng:
            idxs = (frame_ids == f_id)
            if seq_indices is not None:
                idxs = idxs & (seq_indices.squeeze() == seq).numpy()  # indices for the same person id
            idxs = np.where(idxs)
            sorted_idxs = idxs[0][np.argsort(frame_ids[idxs])]  # sort indices by frame_id
            plt.scatter(
                location_pred[sorted_idxs, 0],
                location_pred[sorted_idxs, 1],
                label=f"Prediction {person_id}",
                color=colors[i],
                s=3
                )

    plt.xlabel("Xw")
    plt.ylabel("Yw")
    plt.title("Person Positions")
    plt.ylim(-500, 2250)
    plt.xlim(-255, 1000)
    import os

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plt.savefig(f"{savepath}/{datetime.datetime.now()}_{name}.pdf")

def plot_wildtrack_detections(bbox_pred, bbox_gt, y_out, y_seq,name:str="",seq_indices=None,seq_indices_gt=None,movie=False,newids=None,annotate=False):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 12))
    #ipdb.set_trace()
    location_pred = bbox_pred[:, 9:11]  # x,y in world coordinates
    location_gt = bbox_gt[:, 9:11]  # x,y in world coordinates
    # keep_ids = y_out[:,1]!=-1
    # y_out = y_out[keep_ids]
    # y_seq = y_seq[0][keep_ids]
    id_pred = y_out[:, 1]
    id_gt = y_seq[:, 1]
    frame_ids = y_seq[:, 0]  # frame id's
    from ipdb import set_trace
    #set_trace()
    rng = [0]
    if seq_indices is not None:
        rng = torch.unique(seq_indices)        
    
    for i, person_id in enumerate(np.unique(id_gt)):
        for seq in rng:
            if person_id == -1:
                continue
            idxs = (id_gt == person_id) & (frame_ids >= 360)
            if seq_indices_gt is not None:
                idxs = idxs & (seq_indices_gt.squeeze() == seq).numpy()  # indices for the same person id
            elif seq_indices is not None:
                idxs = idxs & (seq_indices.squeeze() == seq).numpy()
            idxs = np.where(idxs)
            sorted_idxs = idxs[0][np.argsort(frame_ids[idxs])]  # sort indices by frame_id
            plt.plot(
                location_gt[sorted_idxs, 0],
                location_gt[sorted_idxs, 1],
                "o-",
                label=f"Person {person_id}",
                linewidth=3,
                alpha=0.05,
                color="black",
                markersize=3
            )
            
        

    cmap = plt.get_cmap("tab20")

    frame_ids = y_out[:, 0]
    x = len(np.unique(id_pred))
    # Generate an array of equally spaced values between 0 and 1
    values = np.linspace(0, 1, x)

    # Use a colormap to get a color for each value
    colormap = plt.cm.viridis  # you can change this to other colormaps like plasma, inferno, etc.
    colors = [colormap(val) for val in values]
    name = f"{datetime.datetime.now()}_{name}"
    if not os.path.exists(f"viz/graphs/{name}/"):
        os.makedirs(f"viz/graphs/{name}/")
    plt.tight_layout()


    if movie:
        pfr = np.unique(frame_ids)
    else:
        pfr = [np.unique(frame_ids)[-1]]
    
    traj_ids = np.unique(id_pred)
    if newids is not None:
        traj_ids = np.unique(newids)
        if annotate:
            for idx,newid in enumerate(newids):
                if newid == -1:
                    continue
                x,y = location_pred[idx]
                plt.annotate(f"{int(newid)}", (x,y),alpha=0.8,size=4)

    for frid in tqdm(pfr,desc="frame",position=0):
        for i, person_id in tqdm(enumerate(traj_ids),desc="person",position=1,leave=False):
            for seq in tqdm(rng,desc="seq",position=2,leave=False):
                if person_id == -1:
                    continue
                idxs = (id_pred == person_id) & (frame_ids >= 360)
                if movie:
                    idxs = idxs & ((frame_ids == frid) | (frame_ids == frid-1))
                if seq_indices is not None:
                    idxs = idxs & (seq_indices.squeeze() == seq).numpy()  # indices for the same person id
                idxs = np.where(idxs)
                #set_trace()
                sorted_idxs = idxs[0][np.argsort(frame_ids[idxs])]  # sort indices by frame_id
                if isinstance(sorted_idxs, int) or isinstance(sorted_idxs, np.int64):
                    n = 1
                else:
                    n = len(sorted_idxs)
                plt.plot(
                    location_pred[sorted_idxs, 0],
                    location_pred[sorted_idxs, 1],
                    "x-",
                    label=f"Prediction {person_id}",
                    color=colors[i],#cmap(i % 20),
                    linewidth=1,
                    markersize=2
                )
        if movie:
            print(f"Saving frame {frid}")
            plt.savefig(f"viz/graphs/{name}/{frid}.jpg")
    print("Saving final figure... ",end="")
    plt.ylim(-500, 2250)
    plt.xlim(-255, 1000)
    plt.savefig(f"viz/graphs/{name}/final.pdf")
    plt.close()
    print("Done!")

def plot_wildtrack_detections_clustered(bbox_pred, bbox_gt, y_out, y_seq,name:str="",seq_indices=None,seq_indices_gt=None,movie=False):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 12))
    #ipdb.set_trace()
    location_pred = bbox_pred[:, 9:11]  # x,y in world coordinates
    location_gt = bbox_gt[:, 9:11]  # x,y in world coordinates
    # keep_ids = y_out[:,1]!=-1
    # y_out = y_out[keep_ids]
    # y_seq = y_seq[0][keep_ids]
    id_pred = y_out[:, 1]
    id_gt = y_seq[:, 1]
    frame_ids = y_seq[:, 0]  # frame id's
    from ipdb import set_trace
    #set_trace()
    rng = [0]
    if seq_indices is not None:
        rng = torch.unique(seq_indices)        
    
    for i, person_id in enumerate(np.unique(id_gt)):
        for seq in rng:
            if person_id == -1:
                continue
            idxs = (id_gt == person_id) & (frame_ids >= 360)
            if seq_indices_gt is not None:
                idxs = idxs & (seq_indices_gt.squeeze() == seq).numpy()  # indices for the same person id
            elif seq_indices is not None:
                idxs = idxs & (seq_indices.squeeze() == seq).numpy()
            idxs = np.where(idxs)
            sorted_idxs = idxs[0][np.argsort(frame_ids[idxs])]  # sort indices by frame_id
            plt.plot(
                location_gt[sorted_idxs, 0],
                location_gt[sorted_idxs, 1],
                "o-",
                label=f"Person {person_id}",
                linewidth=3,
                alpha=0.05,
                color="black",
                markersize=3
            )

    cmap = plt.get_cmap("tab20")

    frame_ids = y_out[:, 0]
    x = len(np.unique(id_pred))
    # Generate an array of equally spaced values between 0 and 1
    values = np.linspace(0, 1, x)

    # Use a colormap to get a color for each value
    colormap = plt.cm.viridis  # you can change this to other colormaps like plasma, inferno, etc.
    colors = [colormap(val) for val in values]
    name = f"{datetime.datetime.now()}_{name}"
    if not os.path.exists(f"viz/graphs/{name}/"):
        os.makedirs(f"viz/graphs/{name}/")
    plt.tight_layout()


    if movie:
        pfr = np.unique(frame_ids)
    else:
        pfr = [np.unique(frame_ids)[-1]]
    for frid in tqdm(pfr,desc="frame",position=0):
        for i, person_id in tqdm(enumerate(np.unique(id_pred)),desc="person",position=1,leave=False):
            if person_id == -1:
                continue
            idxs = (id_pred == person_id) & (frame_ids >= 360)
            if movie:
                idxs = idxs & ((frame_ids == frid) | (frame_ids == frid-1))
            idxs = np.where(idxs)
                
            sorted_idxs = idxs[0][np.argsort(frame_ids[idxs])]  # sort indices by frame_id
            if isinstance(sorted_idxs, int) or isinstance(sorted_idxs, np.int64):
                n = 1
            else:
                n = len(sorted_idxs)
            plt.plot(
                location_pred[sorted_idxs, 0],
                location_pred[sorted_idxs, 1],
                "x-",
                label=f"Prediction {person_id}",
                color=colors[i],#cmap(i % 20),
                linewidth=1,
                markersize=2
            )
        if movie:
            print(f"Saving frame {frid}")
            plt.savefig(f"viz/graphs/{name}/{frid}.jpg")
    print("Saving final figure... ",end="")
    plt.ylim(-500, 2250)
    plt.xlim(-255, 1000)
    plt.savefig(f"viz/graphs/{name}/final.pdf")
    plt.close()
    print("Done!")

def plot_graph_pyg_preds_mv_view(graph:Data,data_out: Data, seq:int=None,name:str="",ignore_negative:bool=True,edge_type:str="temporal",annotate:bool=False, bbox_pred=None, bbox_gt=None):
    view_labels= get_labels_view_bool(graph)
    from ipdb import set_trace
    if edge_type not in ["temporal","view"]:
        raise NotImplementedError
    import matplotlib.pyplot as plt
    if seq is not None and "seq_indices_node" in graph.keys:
        from utils.multiview_helper import extract_sequence
        data = extract_sequence(graph,seq)
        data_out = extract_sequence(data_out,seq)
    else:
        data = graph
    assert data.x.shape[1] == 9, "Make sure to include spatial features!"

    if bbox_pred is not None:
        location_pred = bbox_pred[0, :, 2:].detach().cpu().numpy().astype("float32")[:, 9:11]  # x,y in world coordinates
        obj_info = bbox_pred[0, :, :2]
    if bbox_gt is not None:
        location_gt = bbox_gt[:, 9:11]  # x,y in world coordinates

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(30, 20), dpi=120)



    for i, feat in enumerate(data.y_pred):
        _seq = data.seq_indices_node[i].item()
        y_out_idx = feat[1].item() + (graph.seq_indices_y_out == _seq).nonzero()[0].item()

        timestamp = obj_info[y_out_idx,0].long().item()
        objid = obj_info[y_out_idx,1].long().item()

        viewid = f"v{data.seq_indices_node[i].item()}" if "seq_indices_node" in data.keys else ""
        obj_name = f"t{timestamp}id{objid}{viewid}"


        x, y = location_pred[y_out_idx]

        alpha_val = 1
        plt.scatter(x, y, alpha=alpha_val, label=f"Detection",color="orange")
        if annotate:
            plt.annotate(obj_name, (x,y),alpha=0.8,size=10)#,bbox = dict(boxstyle="round", fc="0.1"))
    
    edge_index = data.edge_index_v.cpu().numpy()

    for e_i, e in enumerate(edge_index.T):
        if edge_type == "view":
            is_traj = view_labels[e_i].item()
            color = "green" if is_traj else "red"
            marker = "-"

        score = round(data_out.y_e_v_act[e_i, 0].item() ,3)
        if score < 0.5:
            if is_traj:
                color = "blue"
            else:
                color = "yellow"
        if score <0.2:
            continue

        #START PLOTGT
        # if not is_traj:
        #     continue
        # score=1
        #END PLOTGT

        src, dst= e
        seq_src, seq_dst = data.seq_indices_node[src].item(), data.seq_indices_node[dst].item()
        y_out_src_idx = data.y_pred[src, 1].item() + (graph.seq_indices_y_out == seq_src).nonzero()[0].item()
        y_out_dst_idx = data.y_pred[dst, 1].item() + (graph.seq_indices_y_out == seq_dst).nonzero()[0].item()
        timestamp_src = obj_info[y_out_src_idx,0].long().item()
        timestamp_dst = obj_info[y_out_dst_idx,0].long().item()
        x_s, y_s = location_pred[y_out_src_idx]
        x_d, y_d = location_pred[y_out_dst_idx]
        plt.plot([x_s, x_d], [y_s, y_d], marker,color=color, alpha=score)

    seq_str = "" if seq is None else f"_seq{seq}"

    plt.xlabel("Xw")
    plt.ylabel("Yw")
    plt.title("Person Positions")
    plt.ylim(0, 2000)
    plt.xlim(0, 1000)
    savepath = f"viz/graphs/{args.exp_name}"
    import os

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plt.savefig(f"{savepath}/{datetime.datetime.now()}_{name}{seq_str}_{edge_type}.pdf")
    #plt.close()



def plot_graph_pyg_preds_mv_social(graph:Data,data_out: Data, seq:int=None,name:str="",ignore_negative:bool=True,edge_type:str="temporal",annotate:bool=False, bbox_pred=None, bbox_gt=None):
    view_labels= get_labels_view_bool(graph)
    from ipdb import set_trace
    import matplotlib.pyplot as plt
    if seq is not None and "seq_indices_node" in graph.keys:
        from utils.multiview_helper import extract_sequence
        data = extract_sequence(graph,seq)
        data_out = extract_sequence(data_out,seq)
    else:
        data = graph
    assert data.x.shape[1] == 9, "Make sure to include spatial features!"

    if bbox_pred is not None:
        location_pred = bbox_pred[0, :, 2:].detach().cpu().numpy().astype("float32")[:, 9:11]  # x,y in world coordinates
        obj_info = bbox_pred[0, :, :2]
    if bbox_gt is not None:
        location_gt = bbox_gt[:, 9:11]  # x,y in world coordinates

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(30, 20), dpi=120)

    for i, feat in enumerate(data.y_pred):
        _seq = data.seq_indices_node[i].item()
        y_out_idx = feat[1].item() + (graph.seq_indices_y_out == _seq).nonzero()[0].item()

        timestamp = obj_info[y_out_idx,0].long().item()
        objid = obj_info[y_out_idx,1].long().item()

        viewid = f"v{data.seq_indices_node[i].item()}" if "seq_indices_node" in data.keys else ""
        obj_name = f"t{timestamp}id{objid}{viewid}"


        x, y = location_pred[y_out_idx]

        alpha_val = 1
        plt.scatter(x, y, alpha=alpha_val, label=f"Detection",color="orange")
        if annotate:
            plt.annotate(obj_name, (x,y),alpha=0.8,size=10)#,bbox = dict(boxstyle="round", fc="0.1"))
    
    edge_index = data.edge_index_s.cpu().numpy()

    for e_i, e in enumerate(edge_index.T):
        marker = "-"
        color = "green"
        #START PLOTGT
        # if not is_traj:
        #     continue
        # score=1
        #END PLOTGT

        src, dst= e
        seq_src, seq_dst = data.seq_indices_node[src].item(), data.seq_indices_node[dst].item()
        y_out_src_idx = data.y_pred[src, 1].item() + (graph.seq_indices_y_out == seq_src).nonzero()[0].item()
        y_out_dst_idx = data.y_pred[dst, 1].item() + (graph.seq_indices_y_out == seq_dst).nonzero()[0].item()
        timestamp_src = obj_info[y_out_src_idx,0].long().item()
        timestamp_dst = obj_info[y_out_dst_idx,0].long().item()
        x_s, y_s = location_pred[y_out_src_idx]
        x_d, y_d = location_pred[y_out_dst_idx]
        plt.plot([x_s, x_d], [y_s, y_d], marker,color=color, alpha=0.5)

    seq_str = "" if seq is None else f"_seq{seq}"

    plt.xlabel("Xw")
    plt.ylabel("Yw")
    plt.title("Person Positions")
    plt.ylim(0, 2000)
    plt.xlim(0, 1000)
    savepath = f"viz/graphs/{args.exp_name}"
    import os

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plt.savefig(f"{savepath}/{datetime.datetime.now()}_{name}{seq_str}_{edge_type}.pdf")

def plot_graph_pyg_preds_mv(graph:Data,data_out: Data, seq:int=None,name:str="",ignore_negative:bool=True,edge_type:str="temporal",annotate:bool=True, bbox_pred=None, bbox_gt=None):
    from ipdb import set_trace
    #set_trace()
    
    if edge_type not in ["temporal","view"]:
        raise NotImplementedError
    
    import matplotlib.pyplot as plt
    if seq is not None and "seq_indices_node" in graph.keys:
        from utils.multiview_helper import extract_sequence
        data = extract_sequence(graph,seq)
        data_out = extract_sequence(data_out,seq)
    else:
        data = graph

    assert data.x.shape[1] == 9, "Make sure to include spatial features!"
    
    

    if bbox_pred is not None:
        location_pred = bbox_pred[0, :, 2:].detach().cpu().numpy().astype("float32")[:, 9:11]  # x,y in world coordinates
        obj_info = bbox_pred[0, :, :2]
    if bbox_gt is not None:
        location_gt = bbox_gt[:, 9:11]  # x,y in world coordinates

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(30, 20), dpi=120)



    for i, feat in enumerate(data.y_pred):
        _seq = data.seq_indices_node[i].item()
        y_out_idx = feat[1].item() + (graph.seq_indices_y_out == _seq).nonzero()[0].item()

        timestamp = obj_info[y_out_idx,0].long().item()
        objid = obj_info[y_out_idx,1].long().item()

        viewid = f"v{data.seq_indices_node[i].item()}" if "seq_indices_node" in data.keys else ""
        obj_name = f"t{timestamp}id{objid}{viewid}"


        x, y = location_pred[y_out_idx]

        alpha_val = 1
        plt.scatter(x, y, alpha=alpha_val, label=f"Detection",color="orange")
        if annotate:
            plt.annotate(obj_name, (x,y),alpha=0.8,size=10)#,bbox = dict(boxstyle="round", fc="0.1"))
    
    if edge_type == "temporal":
        edge_index = data.edge_index.cpu().numpy() 
    elif edge_type == "view":
        raise NotImplementedError
        edge_index = data.y_e_v_gt.cpu().numpy()

    for e_i, e in enumerate(edge_index.T):
        _seq = data.seq_indices_edge[e_i].item()
        if edge_type == "temporal":
            is_traj = data.labels_e[e_i].item()
            if ignore_negative and not is_traj:
                continue
            color = "green" if is_traj else "red"
            marker = "-x"
        elif edge_type == "view":
            color = "blue"# if is_traj else "red"
            marker = "-"

        score = round(data_out.y_e_act[e_i, 0].item() ,3)
        if score < 0.5:
            if is_traj:
                color = "blue"
            else:
                color = "yellow"
        if score <0.1:
            continue

        src, dst= e
        y_out_src_idx = data.y_pred[src, 1].item()  + (graph.seq_indices_y_out == _seq).nonzero()[0].item()
        y_out_dst_idx = data.y_pred[dst, 1].item()  + (graph.seq_indices_y_out == _seq).nonzero()[0].item()
        timestamp_src = obj_info[y_out_src_idx,0].long().item()
        timestamp_dst = obj_info[y_out_dst_idx,0].long().item()
        x_s, y_s = location_pred[y_out_src_idx]
        x_d, y_d = location_pred[y_out_dst_idx]
        if (x_s > 0 and y_s > 0) or (x_d > 0 and y_d > 0):
            plt.plot([x_s, x_d], [y_s, y_d], marker,color=color, alpha=score)

    seq_str = "" if seq is None else f"_seq{seq}"

    plt.xlabel("Xw")
    plt.ylabel("Yw")
    plt.title("Person Positions")
    plt.ylim(0, 2000)
    plt.xlim(0, 1000)
    savepath = f"viz/graphs/{args.exp_name}"
    import os

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plt.savefig(f"{savepath}/{datetime.datetime.now()}_{name}{seq_str}_{edge_type}.pdf")
    #plt.close()


def plot_graph_pyg_preds(graph:Data,data_out: Data, seq:int=None,name:str="",ignore_negative:bool=True,edge_type:str="temporal",annotate:bool=True, bbox_pred=None, bbox_gt=None):
    from ipdb import set_trace
    #set_trace()
    
    if edge_type not in ["temporal","view"]:
        raise NotImplementedError
    
    import matplotlib.pyplot as plt
    if seq is not None and "seq_indices_node" in graph.keys:
        from utils.multiview_helper import extract_sequence
        data = extract_sequence(graph,seq)
        data_out = extract_sequence(data_out,seq)
    else:
        data = graph

    assert data.x.shape[1] == 9, "Make sure to include spatial features!"
    
    

    if bbox_pred is not None:
        location_pred = bbox_pred[0, :, 2:].detach().cpu().numpy().astype("float32")[:, 9:11]  # x,y in world coordinates
        obj_info = bbox_pred[0, :, :2]
    if bbox_gt is not None:
        location_gt = bbox_gt[:, 9:11]  # x,y in world coordinates

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(30, 20), dpi=120)

    if edge_type == "temporal":
        edge_index = data.edge_index.cpu().numpy() 
    elif edge_type == "view":
        raise NotImplementedError
        edge_index = data.y_e_v_gt.cpu().numpy()
    # Create a dictionary to store color and last location for each person

    for i, feat in enumerate(data.y_pred):
        #_seq = data.seq_indices_node[i].item()
        y_out_idx = feat[1].item() #+ (graph.seq_indices_y_out == _seq).nonzero()[0].item()

        timestamp = obj_info[y_out_idx,0].long().item()
        objid = obj_info[y_out_idx,1].long().item()

        viewid = f"v{data.seq_indices_node[i].item()}" if "seq_indices_node" in data.keys else ""
        obj_name = f"t{timestamp}id{objid}{viewid}"

        x, y = location_pred[y_out_idx]

        alpha_val = 1
        plt.scatter(x, y, alpha=alpha_val, label=f"Detection",color="orange")
        if annotate:
            plt.annotate(obj_name, (x,y),alpha=0.8,size=10)#,bbox = dict(boxstyle="round", fc="0.1"))
    
    for e_i, e in enumerate(edge_index.T):
        #_seq = data.seq_indices_edge[e_i].item()
        if edge_type == "temporal":
            is_traj = data.labels_e[e_i].item()
            if ignore_negative and not is_traj:
                continue
            color = "green" if is_traj else "red"
            marker = "-x"
        elif edge_type == "view":
            # is_traj = data.labels_e_v_gt[i].item()
            # if ignore_negative and not is_traj:
            #     continue
            color = "blue"# if is_traj else "red"
            marker = "-"


        score = round(data_out.y_e_act[e_i, 0].item() ,3)
        if score <0.1:
            continue

        src, dst= e

        y_out_src_idx = data.y_pred[src, 1].item()  #+ (graph.seq_indices_y_out == _seq).nonzero()[0].item()
        y_out_dst_idx = data.y_pred[dst, 1].item()  #+ (graph.seq_indices_y_out == _seq).nonzero()[0].item()

        timestamp_src = obj_info[y_out_src_idx,0].long().item()
        timestamp_dst = obj_info[y_out_dst_idx,0].long().item()
        #set_trace()
        # feat_src = feats[src]
        # feat_dst = feats[dst]

        # edge_feat = (feat_dst + feat_src) / 2
        # x_e, y_e = edge_feat[6], edge_feat[7]
        
        x_s, y_s = location_pred[y_out_src_idx]
        x_d, y_d = location_pred[y_out_dst_idx]
        if (x_s > 0 and y_s > 0) or (x_d > 0 and y_d > 0):
            # plt.scatter(x_e, y_e, alpha=alpha_val, label=f"Edge")
            plt.plot([x_s, x_d], [y_s, y_d], marker,color=color, alpha=score)
        # else:
            #print(f"skipping edge x_s:{x_s:.2f}, y_s:{y_s:.2f}, x_d:{x_d:.2f}, y_d:{y_d:.2f}")

    seq_str = "" if seq is None else f"_seq{seq}"

    plt.xlabel("Xw")
    plt.ylabel("Yw")
    plt.title("Person Positions")
    plt.ylim(0, 2000)
    plt.xlim(0, 1000)

    savepath = f"viz/graphs/{args.exp_name}"
    import os

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plt.savefig(f"{savepath}/{datetime.datetime.now().strftime('%m-%d')}_{name}{seq_str}_{edge_type}.pdf")
    #plt.close()


def plot_trajectories(gt_triplet, pred_triplet, show_ids=True, pdf_filename=None):
    gt_points, gt_ids, gt_timestamps = gt_triplet
    pred_points, pred_ids, pred_timestamps = pred_triplet

    gt_points = gt_points.numpy()[:, :2]  # Only take x and y coordinates
    pred_points = pred_points.numpy()[:, :2]  # Only take x and y coordinates

    # Create a 2D plot
    fig, ax = plt.subplots(figsize=(20, 20))

    # Plot ground truth trajectories
    for id in np.unique(gt_ids):
        mask = gt_ids == id
        points = gt_points[mask]
        timestamps = gt_timestamps[mask]
        color = get_color_for_instance(id)
        
        # Sort points by timestamp
        sorted_indices = np.argsort(timestamps)
        sorted_points = points[sorted_indices]
        
        # Ensure sorted_points is always 2D
        if len(sorted_points.shape) == 1:
            sorted_points = sorted_points.reshape(1, -1)
        
        ax.plot(sorted_points[:, 0], sorted_points[:, 1], c=color, linewidth=2, alpha=0.7)
        ax.scatter(sorted_points[:, 0], sorted_points[:, 1], c=[color], s=30, alpha=0.7)
        if show_ids:
            ax.text(sorted_points[0, 0], sorted_points[0, 1], f'GT:{id}', fontsize=10, ha='right', va='bottom', color=color, alpha=0.5)

    # Plot predicted trajectories
    for id in np.unique(pred_ids):
        mask = pred_ids == id
        points = pred_points[mask]
        timestamps = pred_timestamps[mask]
        color = get_color_for_instance(id)
        
        # Sort points by timestamp
        sorted_indices = np.argsort(timestamps)
        sorted_points = points[sorted_indices]
        
        # Ensure sorted_points is always 2D
        if len(sorted_points.shape) == 1:
            sorted_points = sorted_points.reshape(1, -1)
        
        ax.plot(sorted_points[:, 0], sorted_points[:, 1], c=color, linestyle='--', linewidth=2, alpha=0.7)
        ax.scatter(sorted_points[:, 0], sorted_points[:, 1], c=[color], s=30, marker='s', alpha=0.7)
        if show_ids:
            ax.text(sorted_points[0, 0], sorted_points[0, 1], f'Pred:{id}', fontsize=10, ha='left', va='top', color=color, alpha=0.7)

    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    plt.title('Ground Truth vs Predicted Trajectories', fontsize=16)
    
    # Set aspect ratio to be equal
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.show()
    
    if pdf_filename:
        pdf_filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
        log.info(f"Trajectories visualization saved as {pdf_filename}")
        plt.close()
        return None
    else:
        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return image