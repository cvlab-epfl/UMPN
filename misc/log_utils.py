import collections
import datetime
import logging
import os
import sys
import time

import logging.config
import os
import yaml

import coloredlogs
import verboselogs
from pathlib import Path
from pprint import pformat
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from configs.pathes import conf_path

def get_logging_dict():
    config_dict = None

    if os.path.exists(conf_path["LOGGING_YAML"]):
        with open(conf_path["LOGGING_YAML"], 'rt') as f:
            config_dict = yaml.safe_load(f.read())
    else:
        print(f"Logging configuration file not found at {conf_path['LOGGING_YAML']}")

    return config_dict

def setup_logging(config_dict, default_level=logging.DEBUG):
    """Setup logging configuration

    """
    verboselogs.install()
    coloredlogs.install()

    if config_dict is not None:
        logging.config.dictConfig(config_dict)
    else:
        logging.basicConfig(level=default_level)

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)

        module = os.getcwd().split("/")[-1]
        path_name = record.pathname.split("/")

        if module in path_name:
            id_mod = path_name.index(module)
            record.shortpath = "/".join(path_name[id_mod:])
        else:
            record.shortpath = module + "/" + path_name[-1]

        record.shortpath = ("..."+record.shortpath[-35:]) if len(record.shortpath) > 35 else record.shortpath
        
        return record

    logging.setLogRecordFactory(record_factory)

setup_logging(get_logging_dict())
log = logging.getLogger('pose_estimation')  # this is the global logger


def excepthook(type_, value, traceback):
    import traceback as tb
    log.error("Uncaught exception:\n%s", ''.join(tb.format_exception(type_, value, traceback)))
    # call the default excepthook
    sys.__excepthook__(type_, value, traceback)

sys.excepthook = excepthook


def set_log_level(log_level_name):
    numeric_level = getattr(logging, log_level_name.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    for handler in log.handlers:
        handler.setLevel(numeric_level)

def add_log_file_handler(logfile_path):
    """Add a file handler to the global logger using the configuration from get_logging_dict().

    Args:
        logfile_path (str): Path to the log file.

    Returns:
        logging.FileHandler: The newly created file handler.
    """
    Path(logfile_path).parent.mkdir(exist_ok=True, parents=True)
    config_dict = get_logging_dict()
    file_handler = logging.FileHandler(logfile_path)
    
    if 'handlers' in config_dict and 'file_handler' in config_dict['handlers']:
        file_handler_config = config_dict['handlers']['file_handler']
        formatter_name = file_handler_config.get('formatter', 'simple')
        if 'formatters' in config_dict and formatter_name in config_dict['formatters']:
            formatter_config = config_dict['formatters'][formatter_name]
            formatter = logging.Formatter(fmt=formatter_config.get('format'), datefmt=formatter_config.get('datefmt'))
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.getLevelName(file_handler_config.get('level', 'DEBUG')))
        file_handler.encoding = file_handler_config.get('encoding', 'utf8')
    else:
        # Fallback to default configuration if not found in config_dict
        formatter = logging.Formatter("%(asctime)s.%(msecs)03d - [%(shortpath)38s:%(lineno)03d] - %(levelname)-8s - %(message)s")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        file_handler.encoding = 'utf8'
    
    log.addHandler(file_handler)
    return file_handler


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = -1
        self.avg = -1
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        #TODO add running std to help see convergence

class TimeMeter(AverageMeter):
    def __init__(self):
        super().__init__()
        self.time = time.time()

    def update(self):
        old_time = self.time
        self.time = time.time()
        super().update(self.time - old_time)

class DictMeter(object):
    """ 
        Wrapper function used to track multiple metric stored inside dictionnary
    """

    def __init__(self):
        self.dict_meter = collections.defaultdict(AverageMeter)

    def update(self, dict_metric, n=1):
        for k, v in dict_metric.items():
            self.dict_meter[k].update(v, n=n)

    def avg(self):
        avg_dict = dict()
        for k, v in self.dict_meter.items():
            avg_dict[k] = v.avg

        return avg_dict
    
    def sum(self):
        avg_dict = dict()
        for k, v in self.dict_meter.items():
            avg_dict[k] = v.sum

        return avg_dict

    def __getitem__(self, key):
        if key not in self.dict_meter:
            log.spam(f"Trying to query a key which doesn't exist in the Meter dictionary, returning default - value: {key}")

        return self.dict_meter[key]

    def __contains__(self, item):
        return item in self.dict_meter

    def keys(self):
        return self.dict_meter.keys()

def avg_stat_dict(list_of_dict):
    results = collections.defaultdict(int)

    for d in list_of_dict:
        for k, v in d.items():
            results[k] += v / len(list_of_dict)

    return results

def log_iteration_stats(epoch, global_iter, total_iterations, stats_meter, conf, is_train=True, file_only=False):
    log_message = f"{'Train' if is_train else 'Eval'} Epoch: [{epoch}][{global_iter+1}/{total_iterations}]\t"
    log_message += f"Loss: {stats_meter['loss'].avg:.4f}\t"
    
    for metric in conf["training"]["metric_to_print"]:
        if metric in stats_meter:
            log_message += f"{metric}: {stats_meter[metric].avg:.4f}\t"
    
    for loss in conf["training"]["loss_to_print"]:
        if loss in stats_meter:
            log_message += f"{loss}: {stats_meter[loss].avg:.4f}\t"
    
    log_message += f"Batch Time: {stats_meter['batch_time'].avg:.4f}\t"
    log_message += f"Data Time: {stats_meter['data_time'].avg:.4f}\t"
    log_message += f"Model Time: {stats_meter['model_time'].avg:.4f}\t"
    log_message += f"Criterion Time: {stats_meter['criterion_time'].avg:.4f}\t"
    
    if is_train:
        log_message += f"Optim Time: {stats_meter['optim_time'].avg:.4f}\t"
    
    log_message += f"GPU Memory (iter): {stats_meter['iter_gpu_memory'].avg:.2f} GB (last: {stats_meter['iter_gpu_memory'].val:.2f} GB)\t"
    log_message += f"GPU Memory (chunk): {stats_meter['chunk_gpu_memory'].avg:.2f} GB (last: {stats_meter['chunk_gpu_memory'].val:.2f} GB)\t"
    log_message += f"Detections: {stats_meter['num_detections'].avg:.0f} (last: {stats_meter['num_detections'].val:.0f})\t"
    log_message += f"Edges: {stats_meter['num_edges'].avg:.0f} (last: {stats_meter['num_edges'].val:.0f})"
    
    if file_only:
        for handler in log.handlers:
            if isinstance(handler, logging.FileHandler):
                record = logging.LogRecord(
                    name=log.name,
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=log_message,
                    args=(),
                    exc_info=None
                )
                record.shortpath = ""  # Add the missing 'shortpath' attribute
                try:
                    handler.emit(record)
                except Exception as e:
                    log.error(f"Error emitting log record: {e}")
    else:
        log.info(log_message)

def create_progress_bars(total_iterations, epoch):
    bars = {
        'main': tqdm(total=total_iterations, desc=f"Epoch {epoch}", ncols=150),
        'loss': tqdm(total=0, bar_format='{desc}', position=1, desc="Loss"),
        'metrics': tqdm(total=0, bar_format='{desc}', position=2, desc="Metrics"),
        'time': tqdm(total=0, bar_format='{desc}', position=3, desc="Time"),
        'memory': tqdm(total=0, bar_format='{desc}', position=4, desc="Memory")
    }
    return bars

def update_progress_bars(bars, stats_meter, conf):
    longest_key = max(conf["training"]["loss_to_print"] + conf["training"]["metric_to_print"], key=len)
    longest_value = 8
    
    bars['main'].update(1)
    
    loss_str = f"Loss:    {'Loss':>{len(longest_key)}}: {stats_meter['loss'].avg:{longest_value}.4f}  " + "  ".join([f"{k:>{len(longest_key)}}: {stats_meter[k].avg:{longest_value}.2f}" for k in conf["training"]["loss_to_print"]])
    bars['loss'].set_description_str(loss_str)
    
    metrics_str = f"Metrics: " + "  ".join([f"{k:>{len(longest_key)}}: {stats_meter[k].avg:{longest_value}.2f}" for k in conf["training"]["metric_to_print"]])
    bars['metrics'].set_description_str(metrics_str)
    
    time_str = (f"Time:    "
                f"{'Batch':>{len(longest_key)}}: {stats_meter['batch_time'].avg:{longest_value}.2f}  "
                f"{'Data':>{len(longest_key)}}: {stats_meter['data_time'].avg:{longest_value}.2f}  "
                f"{'Model':>{len(longest_key)}}: {stats_meter['model_time'].avg:{longest_value}.2f}  "
                f"{'Crit':>{len(longest_key)}}: {stats_meter['criterion_time'].avg:{longest_value}.2f}  "
                f"{'Optim':>{len(longest_key)}}: {stats_meter['optim_time'].avg:{longest_value}.2f}  ")
                # f"{'Clean':>{len(longest_key)}}: {stats_meter['clean_time'].avg:{longest_value}.2f}")
    bars['time'].set_description_str(time_str)

    # Precompute the lengths for formatting
    iter_gpu_len = len(f"({stats_meter['iter_gpu_memory'].val:.2f}) GB")
    chunk_gpu_len = len(f"({stats_meter['chunk_gpu_memory'].val:.2f}) GB")
    detections_len = len(f"({stats_meter['num_detections'].val:.0f})")
    edges_len = len(f"({stats_meter['num_edges'].val:.0f})")

    memory_str = (f'Memory:  '
                  f'{"GPU (iter)":>{len(longest_key)}}: {stats_meter["iter_gpu_memory"].avg:{longest_value-iter_gpu_len-1}.2f} ({stats_meter["iter_gpu_memory"].val:.2f}) GB  '
                  f'{"GPU (chunk)":>{len(longest_key)}}: {stats_meter["chunk_gpu_memory"].avg:{longest_value-chunk_gpu_len-1}.2f} ({stats_meter["chunk_gpu_memory"].val:.2f}) GB  '
                  f'{"Detections":>{len(longest_key)}}: {stats_meter["num_detections"].avg:{longest_value-detections_len-1}.0f} ({stats_meter["num_detections"].val:.0f})  '
                  f'{"Edges":>{len(longest_key)}}: {stats_meter["num_edges"].avg:{longest_value-edges_len-1}.0f} ({stats_meter["num_edges"].val:.0f})')
    bars['memory'].set_description_str(memory_str)

def log_epoch(logger, log_dict, epoch):
    flatten_log_dict = flatten_dict(log_dict)
    for k, v in flatten_log_dict.items():
        logger.add_scalar(k, v, epoch)


def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


def dict_to_string(dict):
    return pformat(dict)




# def log_epoch(logger, epoch, train_loss, val_loss, lr, batch_train, batch_val, data_train, data_val, recall):
#     logger.add_scalar('Loss/Train', train_loss, epoch)
#     logger.add_scalar('Loss/Val', val_loss, epoch)
#     logger.add_scalar('Learning/Rate', lr, epoch)
#     logger.add_scalar('Learning/Overfitting', val_loss / train_loss, epoch)
#     logger.add_scalar('Time/Train/Batch Processing', batch_train, epoch)
#     logger.add_scalar('Time/Val/Batch Processing', batch_val, epoch)
#     logger.add_scalar('Time/Train/Data loading', data_train, epoch)
#     logger.add_scalar('Time/Val/Data loading', data_val, epoch)
#     logger.add_scalar('Recall/Val/CapRet/R@1', recall[0][0], epoch)
#     logger.add_scalar('Recall/Val/CapRet/R@5', recall[0][1], epoch)
#     logger.add_scalar('Recall/Val/CapRet/R@10', recall[0][2], epoch)
#     logger.add_scalar('Recall/Val/CapRet/MedR', recall[2], epoch)
#     logger.add_scalar('Recall/Val/ImgRet/R@1', recall[1][0], epoch)
#     logger.add_scalar('Recall/Val/ImgRet/R@5', recall[1][1], epoch)
#     logger.add_scalar('Recall/Val/ImgRet/R@10', recall[1][2], epoch)
#     logger.add_scalar('Recall/Val/ImgRet/MedR', recall[3], epoch)



