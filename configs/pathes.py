from pathlib import Path

conf_path = {
    "ROOT_PATH": Path("./"),
    "LOGGING_YAML": Path("./configs/logging.yaml")
}

data_path = {
    'wildtrack_root': Path('./data/wildtrack/'),
    'scout_root': Path('./data/scout/'),
    'MOT17_root': Path('./data/MOT17/'),
    'MOT20_root': Path('./data/MOT20/'),
}

model_path = {
    "osnet_ain_ms_d_c": Path("./pretrained_detectors/osnet_ain_ms_d_c.pth.tar"),
    "config_rtmdet": Path("./pretrained_detectors/rtmdet_l_swin_b_p6_4xb16-100e_coco.py"),
    "checkpoint_rtmdet": Path("./pretrained_detectors/rtmdet_l_swin_b_p6_4xb16-100e_coco-a1486b6f.pth"),
    "config_yolox": Path("./pretrained_detectors/yolox_x_8x8_300e_coco.py"),
    "checkpoint_yolox": Path("./pretrained_detectors/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"),
}
