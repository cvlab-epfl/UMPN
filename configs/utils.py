import operator
import yaml

import argparse
from functools import reduce

from misc.log_utils import log

def args_to_dict(parser, args):

    d = dict({})

    for group in parser._action_groups:
        d[group.title] = {a.dest: getattr(
            args, a.dest, None) for a in group._group_actions}

    return d
    
def convert_yaml_dict_to_arg_list(yaml_dict):

    arg_list = list()
    print(yaml_dict)

    for k,v in yaml_dict.items():
        if v is None:
            continue
        for args, value in v.items():
            arg_list.append(args)
            if type(value) == list:
                arg_list.extend(value)
            elif value is not None:
                arg_list.append(value)

    print(arg_list)
    return arg_list

def read_yaml_file(file_path):
   
    with open(file_path) as file:
        yaml_dict = yaml.load(file, Loader=yaml.FullLoader)

    return yaml_dict


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

def findDiff(d1, d2, path=[]):
    mismatch = list()
    for k in d1:
        if k in d2:
            if type(d1[k]) is dict:
                mismatch.extend(findDiff(d1[k],d2[k], path + [k]))
        else:
            mismatch.append(path + [k])
    
    return mismatch


def fill_dict_with_missing_value(existing_dict, new_dict, verbose=True):
    deprecated_key = findDiff(existing_dict, new_dict)
    
    if verbose and len(deprecated_key) != 0:
        print("Following keys are no longer in new_dict\n\t",["->".join(keys) for keys in deprecated_key])
    
    missing_new_key = findDiff(new_dict, existing_dict)
    
    if verbose and len(missing_new_key) != 0:
        print("Following key were missing and added to existing_dict\n\t", ["->".join(keys) for keys in missing_new_key])
    
    for key in missing_new_key:
        setInDict(existing_dict, key, getFromDict(new_dict, key))
        
    return existing_dict

def aug_tuple(s):
    try:
        augname, prob = s.split(',')
        return augname, float(prob)
    except:
        raise argparse.ArgumentTypeError("Augmentation tuple list is expected as AugType1,prob1 AugType2,prob2 ... AugTypeN,probN, the type and probability are separeted by a comma and each tuple is separated by a space")

class FeatureDictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        feature_dict = {feat[0]: {'is_norm': feat[1], 'hidden_size': feat[2]} for feat in values}
        setattr(namespace, self.dest, feature_dict)
        
def feature_tuple(s):
    try:
        features, norm, hidden_size = s.split(',')
        norm = norm.lower() in ('true', 't', 'yes', 'y', '1')
        return features, norm, int(hidden_size)
    except:
        raise argparse.ArgumentTypeError("Feature tuple list is expected as FeatureType1,Normalized1,hidden_size1 FeatureType2,Normalized2,hidden_size2 ... FeatureTypeN,NormalizedN,hidden_sizeN, where Normalized is a boolean value (true/false, yes/no, 1/0) and hidden_size is an integer. The features are separated by a comma.")


def merge_checkpoint_config(config, checkpoint_config):
    """
    Merge nested checkpoint config with current config, keeping original values for missing keys.
    Args:
        config: Current configuration dictionary (nested)
        checkpoint_config: Configuration dictionary from checkpoint (nested)
    Returns:
        Merged configuration dictionary
    """
    merged_config = config.copy()
    
    # Save training config if override enabled
    training_config = config["training"] if config["main"].get("override_conf") else None
    
    # Only override if specified
    if config["main"].get("override_conf"):
        # Check for missing top-level keys
        missing_top_keys = set(config.keys()) - set(checkpoint_config.keys())
        if missing_top_keys:
            log.warning(f"The following top-level keys are missing from checkpoint config and will keep original values: {missing_top_keys}")
        
        # For each top-level key in checkpoint config
        for key in checkpoint_config.keys():
            if key not in merged_config:
                continue
                
            # If nested dict, check second level
            if isinstance(checkpoint_config[key], dict):
                missing_nested_keys = set(config[key].keys()) - set(checkpoint_config[key].keys())
                if missing_nested_keys:
                    log.warning(f"For section '{key}', the following keys are missing from checkpoint config and will keep original values: {missing_nested_keys}")
                
                # Merge nested dict keeping original values for missing keys
                merged_config[key] = {
                    **config[key],  # Start with all original values
                    **{k: v for k, v in checkpoint_config[key].items() if k in config[key]}  # Override with checkpoint values that exist in original
                }
            else:
                merged_config[key] = checkpoint_config[key]
        
        # Restore training config
        merged_config["training"] = training_config

    # hacky
    if "compute_metric_in_2d" in checkpoint_config["training"]:
        merged_config["training"]["compute_metric_in_2d"] = checkpoint_config["training"]["compute_metric_in_2d"]
        
    return merged_config