# Detection Archive Management Scripts

This directory contains scripts for managing pre-computed detection archives for various datasets used in the codebase.

## Scripts

### 1. `create_detection_archives.py`
Creates compressed archives of pre-computed detections for different datasets.

**Usage:**
```bash
python create_detection_archives.py <dataset_name> <detector1> [<detector2> ...] [--output-dir <dir>]
```

**Arguments:**
- `dataset_name`: One of `wildtrack`, `scout`, `mot17train`, `mot17test`, `mot20train`, `mot20test`
- `detectors`: One or more detector names (e.g., `yolox`, `rtmdet`, `fasterrcnn`)
- `--output-dir`: Optional output directory for archives (default: current directory)

**Available detectors:**
- `yolox`, `yolox_bytetrack`, `yolox_ghost`
- `rtmdet`, `fasterrcnn`
- `mvaug_frame`, `mvaug_ground`
- `DPM`, `SDP`, `FRCNN`

**Examples:**
```bash
# Create archive with YOLOX detections for Wildtrack
python create_detection_archives.py wildtrack yolox

# Create archive with multiple detectors for SCOUT
python create_detection_archives.py scout yolox rtmdet

# Create archive for MOT17 training set
python create_detection_archives.py mot17train yolox --output-dir ./archives/
```

### 2. `download_detection_archives.py`
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
python download_detection_archives.py wildtrack --base-url http://example.com/detections

# Use a local archive file
python download_detection_archives.py scout --archive-path /path/to/scout_detections_yolox.tar.gz

# Use environment variable for base URL
export DETECTION_ARCHIVE_URL=http://example.com/detections
python download_detection_archives.py mot17train

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
3. The archive naming convention is: `<dataset>_detections_<detector1>_<detector2>_....tar.gz`
4. For downloading, you can either:
   - Provide a direct URL with `--base-url`
   - Use a local archive with `--archive-path`
   - Set the `DETECTION_ARCHIVE_URL` environment variable

## Workflow Example

1. **On the server/machine with detections:**
   ```bash
   # Create archives for all detectors you want to share
   python create_detection_archives.py wildtrack yolox rtmdet
   python create_detection_archives.py scout yolox
   python create_detection_archives.py mot17train yolox
   ```

2. **Upload archives to a web server or file sharing service**

3. **On the target machine:**
   ```bash
   # Download and extract detections
   export DETECTION_ARCHIVE_URL=http://your-server.com/detections
   python download_detection_archives.py wildtrack
   python download_detection_archives.py scout
   python download_detection_archives.py mot17train
   ``` 