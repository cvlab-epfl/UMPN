import gc
import glob
import os
import sys

import numpy as np
import torch

from collections import abc
from pathlib import Path
from misc.log_utils import log

if sys.version_info >= (3, 7):
    class NpArray:
        def __class_getitem__(self, arg):
            pass
else:
    # 3.6 and below don't support __class_getitem__
    class _NpArray:
        def __getitem__(self, _idx):
            pass

    NpArray = _NpArray()


def save_checkpoint(model, optimizer, lr_scheduler, conf, epoch_stats, epoch, is_best, save_all=True):
    #torch.save(state, './weights/' + model_name + '_epoch_' + str(epoch) + ".pth.tar")

    project_root = conf["training"]["ROOT_PATH"]
    
    state = {
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "scheduler":lr_scheduler.state_dict(),
        "conf":conf,
        "epoch_stats":epoch_stats,
        "epoch":epoch,
        "is_best":is_best
    }
    # TODO Need to save parameter require grad status
    weights_dir = project_root / "weights"
    if not weights_dir.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)

    file_path = project_root / "weights" / conf["main"]["name"] / (conf["main"]["name"] + '_epoch_' + str(epoch) + ".pth.tar")
    
    #Check if parent dir exist otherwise make it
    file_path.parents[0].mkdir(parents=True, exist_ok=True)
    
    if save_all:
        torch.save(state, str(file_path))

    if is_best:
        torch.save(state, str(project_root / "weights" / conf["main"]["name"] / f"best_{conf['main']['name']}.pth.tar"))

def check_for_existing_checkpoint(project_root, model_name):
    #Check for existing checkpoint coresponding to model name, if some are found we return the one with the largest epoch number

    file_path = str(project_root) + "/weights/" + model_name + "/" + model_name + "_epoch_{epoch_number}.pth.tar"
    
    checkpoint_path_list = glob.glob(file_path.format(epoch_number="*"))

    if len(checkpoint_path_list) == 0:
        return None

    last_epoch = max([int(Path(path).name[:-8].split("_")[-1]) for path in checkpoint_path_list])
    checkpoint = torch.load(file_path.format(epoch_number=last_epoch), map_location='cpu', weights_only=False)

    return checkpoint

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_parameter_require_grad_state(model):
    named_parameter = list(model.named_parameters())
    require_grad_state_dict = {
        name: param.requires_grad for (name, param) in named_parameter}

    return require_grad_state_dict


class PinnableDict(): #collections.UserDict
    def __init__(self, org_data):
        super().__init__()
        self.data = org_data
    
    def __getitem__(self, _idx):
        return self.data[_idx]

    def __setitem__(self, _idx, val):
        self.data[_idx] = val

    def pin_memory(self):
        return PinnableDict(pin_memory(self.data))

    def to(self, *args, **kwargs):
        return PinnableDict(to_device(self.data, *args, **kwargs))

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()
    
    def __str__(self):
        return self._format_dict(self.data)

    def _format_dict(self, d, indent=0):
        output = []
        for key, value in d.items():
            output.append('  ' * indent + str(key) + ':')
            if isinstance(value, dict):
                output.append(self._format_dict(value, indent + 1))
            elif isinstance(value, torch.Tensor):
                output.append('  ' * (indent + 1) + f"Tensor(shape={value.shape}, dtype={value.dtype}, device={value.device})")
            elif isinstance(value, (list, tuple)):
                output.append('  ' * (indent + 1) + self._format_sequence(value, indent + 1))
            else:
                output.append('  ' * (indent + 1) + str(value))
        return '\n'.join(output)

    def _format_sequence(self, seq, indent):
        if len(seq) == 0:
            return '[]'
        elif isinstance(seq[0], dict):
            return '[\n' + ',\n'.join('  ' * (indent + 1) + self._format_dict(item, indent + 2) for item in seq) + '\n' + '  ' * indent + ']'
        elif isinstance(seq[0], (list, tuple)):
            return '[\n' + ',\n'.join('  ' * (indent + 1) + self._format_sequence(item, indent + 1) for item in seq) + '\n' + '  ' * indent + ']'
        else:
            return str(seq)

    def __repr__(self):
        return self.__str__()

    # #Pass undefined attribute/method to data object
    # def __getattr__(self, name):
    #     return getattr(self.data, name)
            


def to_device(data_elements, *args, **kwargs):
    log.spam(f"to device {args} element {data_elements}")
    log.spam(f"Mappping {isinstance(data_elements, abc.Mapping)} sequence {isinstance(data_elements, abc.Sequence)} has_to {callable(getattr(data_elements, 'to', None))} {type(data_elements)}")
    """Given a collection of data element, will transfer to subelement to correct desired location according to args and kwargs parameters """
    if isinstance(data_elements, abc.Mapping):
        return {key: to_device(value, *args, **kwargs) for key, value in data_elements.items()}
    elif isinstance(data_elements, abc.Sequence) and not isinstance(data_elements, str):
        return [to_device(value, *args, **kwargs) for value in data_elements]
    elif callable(getattr(data_elements, "to", None)):
        return data_elements.to(*args, **kwargs)
    else:
        return data_elements


def pin_memory(data_elements):
    """Given a collection of data element, will call pin_memory to subelement to correct desired location according to args and kwargs parameters """
    if isinstance(data_elements, abc.Mapping):
        return {key: pin_memory(value) for key, value in data_elements.items()}
    elif isinstance(data_elements, abc.Sequence) and not isinstance(data_elements, str):
        return [pin_memory(value) for value in data_elements]
    elif callable(getattr(data_elements, "pin_memory", None)):
        return data_elements.pin_memory()
    else:
        return data_elements


def stack_tensors(batch):
    if isinstance(batch, abc.Mapping):
        return {k:stack_tensors(v) for k,v in batch.items()}
    elif isinstance(batch, abc.Sequence) and torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    else:
        return batch
        
def expand_tensors_dim(batch, dim):
        if isinstance(batch, collections.Mapping):
            return {k:expand_tensors_dim(v, dim) for k,v in batch.items()}
        elif torch.is_tensor(batch):
            return batch.unsqueeze(dim)
        else:
            return batch
        
        
def listdict_to_dictlist(batch):
    if len(batch) != 0 and isinstance(batch[0], abc.Mapping):
        return {key: listdict_to_dictlist([d[key] for d in batch]) for key in batch[0]}
    else:
        return batch

def dict_merge(*dicts_list, empty_dict=None):
    if empty_dict is not None:
        result = empty_dict
    else:
        result = {}
        
    for d in dicts_list:
        for k, v in d.items():
            result[k].extend(v)

    return result


def dict_mean(dict_list):
    if len(dict_list) == 0:
        return {}
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]


def actualsize(input_obj):
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size