import torch.nn as nn

def get_mlp(input_dim, output_dim, nhidden=None, nlayer=2, activation=nn.ReLU, use_batch_norm=False, dropout_rate=0.0):
    if nhidden is None:
        nhidden = output_dim

    layers = []
    for i in range(nlayer):
        if i == 0:
            layers.append(nn.Linear(input_dim, nhidden))
        elif i == nlayer - 1:
            layers.append(nn.Linear(nhidden, output_dim))
        else:
            layers.append(nn.Linear(nhidden, nhidden))
        
        if i < nlayer - 1:
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(nhidden))
            else:
                layers.append(nn.LayerNorm(nhidden))
            
            layers.append(activation())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

    return nn.Sequential(*layers)