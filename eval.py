import torch
import numpy as np
import os
import psutil
import time
import statistics
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

import model.factory as model_factory
import loss.factory as loss_factory
import datasets.factory as data_factory

from configs.arguments import get_config_dict
from misc.log_utils import log, dict_to_string, DictMeter, log_iteration_stats, dict_to_string, create_progress_bars, update_progress_bars
from misc.utils import save_checkpoint, actualsize, check_for_existing_checkpoint
from misc.visualization import plot_trajectories
from misc.mot_metric import MOTMetricEvaluator
from graph.graph import HeteroGraph

from configs.utils import merge_checkpoint_config

# training function
def val(val_loaders, model, criterion, epoch, conf):
    stats_meter = DictMeter()
    model.eval()

    total_iterations = sum(len(loader) for loader in val_loaders)

    progress_bars = create_progress_bars(total_iterations, epoch)

    if conf["training"]["compute_metric_in_2d"]:
        mot_metric_evaluator = MOTMetricEvaluator(interpolate_missing_detections=True)
    else:
        mot_metric_evaluator = None
    with torch.no_grad():
        for seq_idx, val_loader in enumerate(val_loaders):
            graph = HeteroGraph(conf["model_conf"], conf["data_conf"], training=False, device=conf["device"])

            start_time = time.time()
            for iter_idx, mv_data in enumerate(val_loader):
                global_iter = sum(len(loader) for loader in val_loaders[:seq_idx]) + iter_idx                
    
                mv_data = mv_data.to(conf["device"])

                data_time = time.time() - start_time

                model_start = time.time()
                graph.update_graph(mv_data, model)

                if not hasattr(graph.data["detection"], "x"):
                    continue
                
                try:
                    model_output = model(graph)
                except Exception as e:
                    import pdb; pdb.set_trace()
                model_time = time.time() - model_start

                criterion_start = time.time()
                loss_dict = criterion(graph)
                criterion_time = time.time() - criterion_start

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
                    log_iteration_stats(epoch, global_iter, total_iterations, stats_meter, conf, is_train=False, file_only=True)
                if global_iter == (total_iterations - 1):
                    log_iteration_stats(epoch, global_iter, total_iterations, stats_meter, conf, is_train=False)

                start_time = time.time()
                
            # Compute metric on full sequence
            # while graph.data['detection'].timestamp.shape[0] > 0:
            #     graph.archive_last_timestep()
            #     model_output = model(graph)
            
            graph.archive_remaining_data()

            metrics = graph.get_metrics(use_archived=True)
            # Log metrics for the full sequence
            log.info(f"Full sequence metrics for sequence {seq_idx}:")
            for metric_name, metric_value in metrics.items():
                log.info(f"  {metric_name}: {metric_value:.4f}")

            full_sequence_metrics = {f"full_sequence_{k}_seq_{seq_idx}": v for k, v in metrics.items()}
            stats_meter.update(full_sequence_metrics)

            full_sequence_avg_metrics = {f"full_sequence_avg_{k}": v for k, v in metrics.items()}
            stats_meter.update(full_sequence_avg_metrics)

            dset_name = val_loader.dataset.dataset.scene_set.dset_name
            sequence = val_loader.dataset.dataset.scene_set.sequence

            tracking_metrics, gt_trajectories, pred_trajectories = graph.get_tracking_metrics(use_archived=True, dset_name=dset_name, sequence=sequence, mot_metric_evaluator=mot_metric_evaluator, metric_threshold=conf["data_conf"]["metric_threshold"])

            if False:
                graph_cost = graph.compute_graph_cost(use_archived=True, type="pred")
                graph_cost_lower_bound = graph.compute_graph_cost(use_archived=True, type="lower_bound") 
                graph_cost_gt = graph.compute_graph_cost(use_archived=True, type="gt")
                
                log.info(f"Graph costs for sequence {dset_name}-{sequence}:")
                log.info(f"  Predicted graph cost: {graph_cost:.4f}")
                log.info(f"  Lower bound graph cost: {graph_cost_lower_bound:.4f}")
                log.info(f"  Optimality gap: {1 - graph_cost / graph_cost_lower_bound:.4f}")         

            if tracking_metrics is not None:
                log.info(f"Tracking metrics for sequence {seq_idx}:")
                log.info(dict_to_string(tracking_metrics))
                
                full_sequence_tracking_metrics = {f"full_sequence_{k}_seq_{seq_idx}": v for k, v in tracking_metrics.items()}
                stats_meter.update(full_sequence_tracking_metrics)

                full_sequence_avg_tracking_metrics = {f"full_sequence_avg_{k}": v for k, v in tracking_metrics.items()}
                stats_meter.update(full_sequence_avg_tracking_metrics)
            # Visualize graph
            if conf["training"]["generate_visualization"]:  
                graph.visualize_graph(use_archived=True, display_social_edges=False, pdf_filename=conf["training"]["ROOT_PATH"] / "viz" / conf["main"]["name"] / f"val_graph_{conf['main']['name']}_seq_{dset_name}-{sequence}_epoch_{epoch}.pdf")
                plot_trajectories(gt_trajectories, pred_trajectories, pdf_filename=conf["training"]["ROOT_PATH"] / "viz" / conf["main"]["name"] / f"val_graph_{conf['main']['name']}_seq_{dset_name}-{sequence}_epoch_{epoch}_trajectories.pdf")
            if conf["training"]["generate_video"]:
                selected_views = val_loader.dataset.dataset.scene_set.get_views()
                graph.visualize_sequence_predictions(conf["training"]["ROOT_PATH"] / "viz" / conf["main"]["name"] / f"val_graph_{conf['main']['name']}_seq_{dset_name}-{sequence}_epoch_{epoch}_predictions.mp4", selected_views, use_archived=True, show_pred=True, show_gt=False)
            if conf["training"]["export_graph"]:
                graph.export_graph_as_json(json_path=conf["training"]["ROOT_PATH"] / "viz" / conf["main"]["name"] / f"val_graph_{conf['main']['name']}_seq_{dset_name}-{sequence}_epoch_{epoch}.json", use_archived=True)
            if conf["training"]["export_gt"]:
                graph.save_gt_dict_to_json(conf["training"]["ROOT_PATH"] / "viz" / conf["main"]["name"] / f"val_gt_dict_{conf['main']['name']}_seq_{dset_name}-{sequence}_epoch_{epoch}.json")

    for bar in progress_bars.values():
        bar.close()

    if mot_metric_evaluator is not None:
        metrics = mot_metric_evaluator.compute_metrics(dset_name)
        log.info(f"Tracking metrics for sequence {seq_idx}:")
        log.info(dict_to_string(metrics))

        full_sequence_tracking_metrics = {f"full_sequence_{k}": v for k, v in metrics.items()}
        stats_meter.update(full_sequence_tracking_metrics)

    return {"stats": stats_meter.avg()}


if __name__ == '__main__':

    # Parse config file
    config = get_config_dict()

    checkpoint_path = config["main"].get("checkpoint")
    if checkpoint_path:
        log.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
        if config["main"].get("override_conf"):
            config = merge_checkpoint_config(config, checkpoint["conf"])
    else:
        log.error("No checkpoint path specified in config[main]")
        exit(1)
    
    log.debug(dict_to_string(config))

    # Set device
    config["device"] = torch.device('cuda' if torch.cuda.is_available() and config["main"]["device"] == "cuda" else 'cpu')
    log.info(f"Device: {config['device']}")

    # Load data
    log.info("Loading Data ...")
    _, val_dataloaders = data_factory.get_dataloader(config["data_conf"])
    log.info("Data loaded")

    # Initialize model
    log.info("Initializing model ...")
    model = model_factory.get_model(config["model_conf"], config["data_conf"])
    model.dynamic_size_initialization(val_dataloaders[0])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(config["device"])
    log.info("Model initialized")

    # Initialize criterion
    criterion = loss_factory.get_loss(config["loss_conf"], config["model_conf"], config["data_conf"])

    # Run evaluation
    log.info("Starting evaluation...")
    eval_results = val(val_dataloaders, model, criterion, checkpoint["epoch"], config)
    log.info("Evaluation complete")

    # Log results
    log.info("Evaluation results:")
    log.info(dict_to_string(eval_results["stats"]))

    log.info('Evaluation complete')