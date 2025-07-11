# Precomputed Detection Archives

This directory contains scripts for managing pre-computed detection archives for various datasets used in the codebase.

## Download Detection Archives

Downloads and extracts pre-computed detection archives for datasets.

**Usage:**
```bash
python download_detection_archives.py <dataset_name> [options]
```

**Arguments:**
- `dataset_name`: One of `wildtrack`, `scout`, `mot17train`, `mot17test`, `mot20train`, `mot20test`
- `--base-url`: Base URL for downloading archives
- `--archive-path`: Path to local archive file (skips download)
- `--list`: List available datasets

**Examples:**
```bash
# Download from a URL
python download_detection_archives.py wildtrack 

# Use a local archive file
python download_detection_archives.py scout --archive-path /path/to/scout_detections_yolox.tar.gz

# List available datasets
python download_detection_archives.py --list
```

## Archive Structure

The archives preserve the original folder structure:

- **Wildtrack/SCOUT**: Single detection directory at root
  ```
  pretrained_detections/
  └── <detector_name>/
      ├── 0_0.npy
      ├── 0_1.npy
      └── ...
  ```

- **MOT17/MOT20**: Multiple subfolders for sequences
  ```
  pretrained_detections/
  └── train/
      ├── MOT17-02-DPM/
      │   └── det/
      │       └── <detector_name>/
      │           └── *.npy
      ├── MOT17-04-DPM/
      │   └── det/
      │       └── <detector_name>/
      │           └── *.npy
      └── ...
  ```

## Notes

1. The scripts use `configs/pathes.py` to locate dataset directories
2. Detection files are stored as `.npy` files containing bounding boxes and scores
3. For MOT17 and MOT20, the YOLOX Ghost detections are taken from the following repository: https://github.com/dvl-tum/GHOST