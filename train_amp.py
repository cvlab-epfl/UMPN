import torch
import numpy as np
import os
import psutil
import time
import statistics
import gc

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils.tensorboard import SummaryWriter
from torch import autocast, GradScaler


from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from eval_amp import val
import model.factory as model_factory
import loss.factory as loss_factory
import datasets.factory as data_factory

from configs.arguments import get_config_dict
from misc.log_utils import log, dict_to_string, DictMeter, log_epoch, log_iteration_stats, create_progress_bars, update_progress_bars
from misc.utils import save_checkpoint, actualsize, check_for_existing_checkpoint
from misc.visualization_utils import save_images_as_video

from graph.graph import HeteroGraph



# if False:
#     graph.get_graph_stats(use_archived=False)
#     log.info(f"----------------------------------")
#     graph.get_graph_stats(use_archived=True)


#     graph.visualize_graph(show_views=None, display_temp_edges=True, display_view_edges=True, start_time=None, end_time=None, pdf_filename=f"viz/graph_seq_{seq_idx}_full_graph_predtemp_{iter_idx*5}.pdf", display_social_edges=True, use_archived=False)
#     if iter_idx < 5:
#         continue
    
#     graph.visualize_graph(show_views=None, display_temp_edges=True, display_view_edges=True, start_time=None, end_time=None, pdf_filename=f"viz/graph_seq_{seq_idx}_full_graph_archived_{iter_idx*5-25}.pdf", display_social_edges=True, use_archived=True)
#     continue

#     # Generate and save graph visualization frame
#     graph_frame = graph.visualize_graph(show_views=None, display_temp_edges=True, display_view_edges=True, start_time=None, end_time=None, pdf_filename=None, display_social_edges=True, use_archived=True)
#     graph_frames.append(graph_frame)
    
#     for view in range(num_views):
#         frame_image_path = f"/cvlabscratch/cvlab/home/engilber/datasets/Wildtrack_dataset/Image_subsets/C{view+1}/{iter_idx*5-25:08d}.png"
#         frame = graph.visualize_single_view(selected_view=view, display_temp_edges=True, display_social_edges=True, pdf_filename=None, last_timestamp=True, frame_image_path=frame_image_path, use_archived=True)
#         frames_per_view[view].append(frame)
    
#     if iter_idx == 20:#len(train_loader) - 1:
#         # Save single view videos
#         for view, frames in frames_per_view.items():
#             video_filename = f"viz/graph_seq_{seq_idx}_view_{view}.mp4"
#             save_images_as_video(frames, video_filename, fps=5)
        
#         # Save graph visualization video
#         graph_video_filename = f"viz/graph_seq_{seq_idx}_full_graph.mp4"
#         save_images_as_video(graph_frames, graph_video_filename, fps=5)
        
#         exit()
    
# training function
def train(train_loaders, model, criterion, optimizer, epoch, conf):
    stats_meter = DictMeter()
    model.train()
    scaler = GradScaler("cuda")

    chunked_train_loaders = [data_factory.chunkify_dataloader(train_loader, conf["data_conf"]["chunk_size"], conf["data_conf"]["overlap_size"]) for train_loader in train_loaders]
    total_iterations = sum(len(loader) for loader, _ in chunked_train_loaders)

    progress_bars = create_progress_bars(total_iterations, epoch)

    for seq_idx, (chunked_train_loader, chunk_lengths) in enumerate(chunked_train_loaders):
        graph = HeteroGraph(conf["model_conf"], conf["data_conf"], training=True, device=conf["device"])
        tot_loss = 0
        current_chunk_length = chunk_lengths.pop(0)

        start_time = time.time()
        for iter_idx, mv_data in enumerate(chunked_train_loader):
            global_iter = sum(len(loader) for loader, _ in chunked_train_loaders[:seq_idx]) + iter_idx               

            mv_data = mv_data.to(conf["device"])
            data_time = time.time() - start_time

            model_start = time.time()
            with autocast("cuda", dtype=torch.bfloat16):
                graph.update_graph(mv_data, model)

                if not hasattr(graph.data["detection"], "x"):
                    continue

                model_output = model(graph)
                
                criterion_start = time.time()
                loss_dict = criterion(graph)
                criterion_time = time.time() - criterion_start

            model_time = time.time() - model_start

            tot_loss += loss_dict["loss"]
            current_chunk_length -= 1

            batch_time = time.time() - start_time

            metrics = graph.get_metrics()
            iter_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # Convert to GB

            epoch_stats_dict = {
                **loss_dict,
                **metrics,
                **graph.get_stats(),
                **model_output.get("time_stats", {}),
                "batch_time": batch_time,
                "data_time": data_time,
                "model_time": model_time,
                "criterion_time": criterion_time,
                "iter_gpu_memory": iter_gpu_memory,
            }
            stats_meter.update(epoch_stats_dict)

            update_progress_bars(progress_bars, stats_meter, conf)

            if global_iter % conf["main"]["print_frequency"] == 0:
                log_iteration_stats(epoch, global_iter, total_iterations, stats_meter, conf, file_only=True)
            if global_iter == (total_iterations - 1):
                log_iteration_stats(epoch, global_iter, total_iterations, stats_meter, conf)

            start_time = time.time()

            # Perform backward pass when current_chunk_length reaches 0 or at the end of the dataloader
            if current_chunk_length == 0 or iter_idx == len(chunked_train_loader) - 1:
                chunk_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # Convert to GB
                scaler.scale(tot_loss).backward()

                if (global_iter + 1) % conf["training"]["accumulate_grad_iter"] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                backward_time = time.time() - start_time
                
                del mv_data
                del graph
                torch.cuda.empty_cache()
                gc.collect()

                clean_time = time.time() - start_time - backward_time

                chunk_size = conf["data_conf"]["chunk_size"] - current_chunk_length
                stats_meter.update({
                    "optim_time": backward_time / chunk_size,
                    "clean_time": clean_time / chunk_size,
                    "chunk_gpu_memory": chunk_gpu_memory
                })

                # Reset total loss and get next chunk length
                tot_loss = 0
                current_chunk_length = chunk_lengths.pop(0) if chunk_lengths else 0
                graph = HeteroGraph(conf["model_conf"], conf["data_conf"], training=True, device=conf["device"])

    for bar in progress_bars.values():
        bar.close()

    return {"stats": stats_meter.avg()}



def compute_is_best(valid_results, best_score):
    current_score = (valid_results["stats"]["node_f1_score"] +
                        valid_results["stats"]["temporal_edge_f1_score"] +
                        valid_results["stats"]["view_edge_f1_score"]) / 3
    is_best = current_score > best_score if best_score is not None else True
    best_score = max(current_score, best_score) if best_score is not None else current_score

    return is_best, best_score

if __name__ == '__main__':

    print("Starting training")
    #parse arg and config file
    config = get_config_dict()
    log.debug(dict_to_string(config))

    ##################
    ### Initialization
    ##################
    logger = SummaryWriter(config["training"]["ROOT_PATH"] / "logs" / config["main"]["name"])

    config["device"] = torch.device('cuda' if torch.cuda.is_available() and config["main"]["device"] == "cuda" else 'cpu') 
    log.info(f"Device: {config['device']}")
    
    start_epoch = 0

    resume_checkpoint = check_for_existing_checkpoint(config["training"]["ROOT_PATH"], config["main"]["name"]) # "model_335")#

    checkpoint_path = config["main"].get("checkpoint")
    if checkpoint_path:
        resume_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if resume_checkpoint is not None:
        log.info(f"Checkpoint for model {config['main']['name']} found, resuming from epoch {resume_checkpoint['epoch']}")
        # assert resume_checkpoint["conf"] == config
        start_epoch = resume_checkpoint["epoch"] + 1

    end = time.time()
    log.info("Loading Data ...")

    train_dataloaders, val_dataloaders = data_factory.get_dataloader(config["data_conf"])
    
    log.info(f"Data loaded in {time.time() - end} s")

    end = time.time()
    log.info("Initializing model ...")

    model = model_factory.get_model(config["model_conf"], config["data_conf"])
    # initialize model deriveg feature size from data and graph
    model.dynamic_size_initialization(train_dataloaders[0])

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["state_dict"])

    model.to(config["device"])

    log.info(f"Model initialized in {time.time() - end} s")

    criterion = loss_factory.get_loss(config["loss_conf"], config["model_conf"], config["data_conf"])

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["training"]["lr"], weight_decay=config["training"]["decay"])
        
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])

    lr_scheduler = MultiStepLR(optimizer, config["training"]["lrd"][1:], gamma=config["training"]["lrd"][0])

    if resume_checkpoint is not None:
        lr_scheduler.load_state_dict(resume_checkpoint["scheduler"])



    ############
    ### Training
    ############
    process = psutil.Process(os.getpid()) 
    
    end = time.time()
    best_score = None
    for epoch in range(start_epoch, config["training"]["max_epoch"]):
        log.info(f"Memory usage {process.memory_info().rss / 1024 / 1024 / 1024} GB")

        log.info(f"{f''' Beginning epoch {epoch} of {config['main']['name']} ''':#^150}")
        with logging_redirect_tqdm(loggers=[log]):
            if not config["model_conf"]["use_oracle"]:
                train_result = train(train_dataloaders, model, criterion, optimizer, epoch, config)
            else:
                train_result = {"stats":{}}
        train_time = time.time() - end
        log.info(f"{f''' Traning for epoch {epoch} of {config['main']['name']} completed in {train_time:.2f}s ''':#^150}")
        
        log.info(f"{f' Beginning validation for epoch {epoch} ':*^150}")
        with logging_redirect_tqdm(loggers=[log]):
            valid_results = val(val_dataloaders, model, criterion, epoch, config)
        log.info(f"{f' Validation for epoch {epoch} completed in {(time.time() - end - train_time):.2f}s ':*^150}")
        if config["model_conf"]["use_oracle"]:
            exit()
        log_epoch_dict = {"train":train_result["stats"], "val":valid_results["stats"], "lr":optimizer.param_groups[0]['lr']}

        is_best, best_score = compute_is_best(valid_results, best_score)

        log_epoch(logger, log_epoch_dict, epoch)
        save_checkpoint(model, optimizer, lr_scheduler, config, log_epoch_dict, epoch, is_best)
       
        lr_scheduler.step()
        log.info(f"Epoch {epoch} completed in {time.time()-end}s")

        end = time.time()
        
    log.info('Training complete')