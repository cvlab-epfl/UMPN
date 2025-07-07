#!/usr/bin/env python3
"""
Download and extract pre-computed detection archives for different datasets.
Usage: python download_detection_archives.py <dataset_name>
"""

import sys
import os
import tarfile
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# Add parent directory to path to import configs
sys.path.append(str(Path(__file__).parent.parent))

from configs.pathes import data_path
from misc.log_utils import log


def get_dataset_root(dataset_name):
    """Get the root path for a dataset."""
    if dataset_name == "wildtrack":
        return data_path['wildtrack_root']
    elif dataset_name == "scout":
        return data_path['scout_root']
    elif dataset_name == "mot17":
        return data_path['MOT17_root']
    elif dataset_name == "mot20":
        return data_path['MOT20_root']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress reporting."""
    try:
        import requests
        
        log.info(f"Downloading from: {url}")
        log.info(f"Destination path: {dest_path}")
        
        # Create destination directory if it doesn't exist
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        log.debug(f"Created directory: {dest_path.parent}")
        
        # Simple high-level download
        log.debug("Making HTTP request...")
        response = requests.get(url)
        log.debug(f"Response status code: {response.status_code}")
        log.debug(f"Response headers: {dict(response.headers)}")
        
        response.raise_for_status()
        
        # Check content length
        content_length = response.headers.get('content-length')
        if content_length:
            log.info(f"File size: {int(content_length) / (1024*1024):.2f} MB")
        else:
            log.warning("Content-Length header not found")
        
        # Write the content to file
        log.debug(f"Writing to file: {dest_path}")
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        
        # Verify file was written
        if dest_path.exists():
            file_size = dest_path.stat().st_size
            log.info(f"Download complete: {dest_path} ({file_size / (1024*1024):.2f} MB)")
        else:
            log.error(f"File was not created: {dest_path}")
            return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        log.error(f"HTTP request failed: {e}")
        log.error(f"Request URL: {url}")
        return False
    except urllib.error.URLError as e:
        log.error(f"Failed to download: {e}")
        return False
    except FileNotFoundError as e:
        log.error(f"File path error: {e}")
        log.error(f"Attempted path: {dest_path}")
        return False
    except PermissionError as e:
        log.error(f"Permission denied: {e}")
        log.error(f"Check write permissions for: {dest_path}")
        return False
    except Exception as e:
        log.error(f"Unexpected error during download: {e}")
        log.error(f"Error type: {type(e).__name__}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        return False


def extract_archive(archive_path, dataset_name):
    """Extract detection archive to the appropriate location."""
    dataset_root = get_dataset_root(dataset_name)
    
    log.info(f"Extracting archive to: {dataset_root}")
    
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # List all members for logging
            members = tar.getmembers()
            log.info(f"Archive contains {len(members)} files/directories")
            
            # Extract all files
            for member in members:                   
                # Extract preserving directory structure
                tar.extract(member, dataset_root)
                
                if member.isfile():
                    log.debug(f"Extracted: {dataset_root / member.name}")
        
        log.info(f"Extraction complete!")
        return True
        
    except Exception as e:
        log.error(f"Failed to extract archive: {e}")
        return False


def get_archive_url(dataset_name, base_url):
    """Construct the archive URL based on dataset name."""
    # This assumes archives are named consistently
    # You can modify this based on your actual file naming convention
    archive_name = f"{dataset_name}_detections_all.tar.gz"
    return f"{base_url}/{archive_name}"


def download_and_extract(dataset_name, base_url=None, archive_path=None):
    """Download and extract detection archive for a dataset."""
    
    # If archive_path is provided, skip download
    if archive_path and Path(archive_path).exists():
        log.info(f"Using local archive: {archive_path}")
        return extract_archive(archive_path, dataset_name)
    
    # Otherwise, download from URL
    if not base_url:
        log.error("No base URL provided and no local archive specified.")
        log.info("Please provide either --base-url or --archive-path")
        return False
    
    # Construct URL and download
    url = get_archive_url(dataset_name, base_url)
    temp_archive = Path(f"/tmp/{dataset_name}_detections.tar.gz")
    
    if download_file(url, temp_archive):
        success = extract_archive(temp_archive, dataset_name)
        
        # # Clean up temporary file
        if temp_archive.exists():
            temp_archive.unlink()
            log.debug(f"Removed temporary archive: {temp_archive}")
        
        return success
    
    return False


def list_available_datasets():
    """List datasets that can be downloaded."""
    datasets = [
        "wildtrack - Wildtrack dataset detections",
        "scout - SCOUT dataset detections", 
        "mot17 - MOT17 dataset detections (both train and test)",
        "mot20 - MOT20 dataset detections (both train and test)"
    ]
    
    log.info("Available datasets:")
    for ds in datasets:
        log.info(f"  {ds}")


def main():
    parser = argparse.ArgumentParser(
        description='Download and extract detection archives for datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from a URL
  python download_detection_archives.py wildtrack --base-url http://example.com/detections
  
  # Use a local archive file
  python download_detection_archives.py scout --archive-path /path/to/scout_detections.tar.gz
  
  # Use environment variable for base URL
  export DETECTION_ARCHIVE_URL=http://example.com/detections
  python download_detection_archives.py mot17
        """
    )
    
    parser.add_argument('dataset', type=str, nargs='?',
                        choices=['wildtrack', 'scout', 'mot17', 'mot20'],
                        help='Dataset name')
    parser.add_argument('--base-url', type=str, default="https://scout.epfl.ch/datas/umpn_data/detection_archives/",
                        help='Base URL for downloading archives (can also use DETECTION_ARCHIVE_URL env var)')
    parser.add_argument('--archive-path', type=str, default=None,
                        help='Path to local archive file (skips download)')
    parser.add_argument('--list', action='store_true',
                        help='List available datasets')
    
    args = parser.parse_args()
    
    if args.list or not args.dataset:
        list_available_datasets()
        return
    
    # Get base URL from args or environment
    base_url = args.base_url or os.environ.get('DETECTION_ARCHIVE_URL')
    
    if not base_url and not args.archive_path:
        log.error("No base URL or archive path provided!")
        log.info("Please provide either:")
        log.info("  1. --base-url argument")
        log.info("  2. --archive-path argument")
        log.info("  3. Set DETECTION_ARCHIVE_URL environment variable")
        return
    
    # Download and extract
    success = download_and_extract(args.dataset, base_url, args.archive_path)
    
    if success:
        log.info(f"Successfully set up detections for {args.dataset}")
    else:
        log.error(f"Failed to set up detections for {args.dataset}")
        sys.exit(1)


if __name__ == "__main__":
    main() 