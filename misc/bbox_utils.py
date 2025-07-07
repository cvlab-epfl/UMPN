import torch


def calculate_iou(box1, box2):
    # Calculate IoU between two bounding boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection / (area1 + area2 - intersection)
    return iou

def calculate_bbox_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        bbox1 (torch.Tensor): Bounding box 1 with shape (N, 4) where each row is [x1, y1, x2, y2]
        bbox2 (torch.Tensor): Bounding box 2 with shape (N, 4) where each row is [x1, y1, x2, y2]
    
    Returns:
        torch.Tensor: IoU values with shape (N,)
    """
    # Calculate intersection coordinates
    x1 = torch.max(bbox1[:, 0], bbox2[:, 0])
    y1 = torch.max(bbox1[:, 1], bbox2[:, 1])
    x2 = torch.min(bbox1[:, 2], bbox2[:, 2])
    y2 = torch.min(bbox1[:, 3], bbox2[:, 3])

    # Calculate intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate union area
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    union = area1 + area2 - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    return iou

def calculate_bbox_ratio(bbox1, bbox2):
    """
    Calculate the ratio of bounding box areas.
    
    Args:
        bbox1 (torch.Tensor): Bounding box 1 with shape (N, 4) where each row is [x1, y1, x2, y2]
        bbox2 (torch.Tensor): Bounding box 2 with shape (N, 4) where each row is [x1, y1, x2, y2]
    
    Returns:
        torch.Tensor: Ratio of areas with shape (N,)
    """
    # Calculate areas
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])

    # Calculate ratio (smaller area / larger area)
    ratio = torch.min(area1, area2) / torch.max(area1, area2)
    return ratio