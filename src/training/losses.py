import torch.nn as nn
def get_loss(name):
    return dict(bce=nn.BCEWithLogitsLoss, mse=nn.MSELoss, huber=nn.HuberLoss)[name]()
