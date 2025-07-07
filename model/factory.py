from model.model import HGnnMPNN, HGnnCameraMPNN
from misc.log_utils import log, dict_to_string


def get_model(model_conf, data_conf):
    log.info(f"Building Model")
    log.debug(f"Model spec: {dict_to_string(model_conf)}")
    # log.debug(f"Data spec: {dict_to_string(data_spec)}")

    if model_conf["use_camera_node"]:
        model = HGnnCameraMPNN(model_conf, data_conf)
    else:
        model = HGnnMPNN(model_conf, data_conf)

    return model