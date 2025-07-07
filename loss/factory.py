from loss.loss import HeteroLoss

def get_loss(loss_spec, model_spec, data_spec):
    return HeteroLoss(loss_spec, model_spec, data_spec)
