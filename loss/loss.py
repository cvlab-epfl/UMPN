import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.log_utils import log


class HeteroLoss(nn.Module):
    def __init__(self, loss_spec, model_spec, data_spec):
        super(HeteroLoss, self).__init__()
        self.loss_spec = loss_spec
        self.model_spec = model_spec
        self.data_spec = data_spec

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(gamma=0, alpha=None, size_average=True)
        # self.focal_loss = FocalLoss(gamma=2.0, alpha=None, size_average=True) #alpha=[0.223, 0.777]
        # self.focal_loss_view = FocalLoss(gamma=3.0, alpha=None, size_average=True) #alpha=[0.986, 0.014]

    def forward(self, graph):
        loss_node = 0
        loss_view = 0
        loss_temporal = 0
        loss_temporal_sigmoid = 0

        # Node classification loss
        if 'pred' in graph.data['detection'] and 'label' in graph.data['detection']:
            if len(graph.data['detection'].pred) > 0:
                loss_node = self.bce_loss(graph.data['detection'].pred, graph.data['detection'].label.unsqueeze(1).float())
                loss_node += compute_loss_focal(graph.data['detection'].pred, graph.data['detection'].label)

        # View edge loss
        if 'edge_pred' in graph.data['detection', 'view', 'detection'] and 'edge_label' in graph.data['detection', 'view', 'detection'] and graph.data['detection', 'view', 'detection'].edge_pred.shape[0] > 0:
            loss_view = self.bce_loss(
                graph.data['detection', 'view', 'detection'].edge_pred,
                graph.data['detection', 'view', 'detection'].edge_label.unsqueeze(1).float()
            )
            loss_view += compute_loss_focal(graph.data['detection', 'view', 'detection'].edge_pred, 
                                            graph.data['detection', 'view', 'detection'].edge_label)
            # view_pred = torch.sigmoid(graph.data['detection', 'view', 'detection'].edge_pred).squeeze()
            # loss_view = self.focal_loss_view(view_pred, graph.data['detection', 'view', 'detection'].edge_label)


        # Temporal edge loss
        if 'edge_pred' in graph.data['detection', 'temporal', 'detection'] and 'edge_label' in graph.data['detection', 'temporal', 'detection'] and graph.data['detection', 'temporal', 'detection'].edge_pred.shape[0] > 0:
            loss_temporal = self.bce_loss(
                graph.data['detection', 'temporal', 'detection'].edge_pred,
                graph.data['detection', 'temporal', 'detection'].edge_label.unsqueeze(1).float()
            )
            loss_temporal += compute_loss_focal(graph.data['detection', 'temporal', 'detection'].edge_pred, 
                                               graph.data['detection', 'temporal', 'detection'].edge_label)

        # Temporal edge loss with sigmoid activation
        if 'edge_pred' in graph.data['detection', 'temporal', 'detection'] and 'edge_label' in graph.data['detection', 'temporal', 'detection'] and graph.data['detection', 'temporal', 'detection'].edge_pred.shape[0] > 0:
            temporal_pred = torch.sigmoid(graph.data['detection', 'temporal', 'detection'].edge_pred).squeeze()
            loss_temporal_sigmoid = self.focal_loss(
                temporal_pred,
                graph.data['detection', 'temporal', 'detection'].edge_label
            )
        
        loss = loss_node + loss_view + loss_temporal + loss_temporal_sigmoid

        return {"loss": loss, 
                "loss_node": loss_node, 
                "loss_view": loss_view, 
                "loss_temporal": loss_temporal,
                "loss_temporal_sigmoid": loss_temporal_sigmoid}
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([1 - alpha, alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.eps = 1e-10

    def forward(self, outputs, targets):
        if len(outputs.shape) == 0:
            outputs = outputs.unsqueeze(0)
        outputs = torch.stack((1 - outputs, outputs), dim=1)  # (N, 2)
        targets = targets.view(-1, 1).to(outputs.device)  # (N, 1)

        logpt = torch.log(outputs + self.eps)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.alpha is not None:
            if self.alpha.type() != outputs.data.type():
                self.alpha = self.alpha.type_as(outputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
        

def focal_loss_binary(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,  # a float
    # alpha: float,
    gamma: float=2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # if alpha >= 0:
    #     alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    #     loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def _compute_focal_loss(final_logits, y, pos_weight, gamma: float):
    return focal_loss_binary(final_logits.view(-1), y.view(-1), pos_weight=pos_weight,
                                gamma=gamma, reduction="none")


def compute_loss_focal(final_class_logits, y, gamma=2):
    pos_weight_multiplier = 0.5
    loss_type ="focal"
    #set_trace()
    pos_count = y.sum()
    pos_weight = ((len(y) - pos_count) / pos_count) * pos_weight_multiplier if pos_count else None

    # TODO: extract focal gamma into a hparam
    if loss_type == "bce":
        pass
        #loss = _compute_bce_loss(final_class_logits, y, pos_weight)
    elif loss_type == "focal":
        loss = _compute_focal_loss(final_class_logits, y.float(), pos_weight, gamma=gamma)
    else:
        raise NotImplementedError(f"Unknown {loss_type} loss")

    loss = loss.mean()
    return loss