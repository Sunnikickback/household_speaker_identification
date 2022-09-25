import torch


def Adam(parameters, lr, weight_decay=0, **kwargs):
    print('Initialised Adam optimizer')
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def SGD(parameters, lr, **kwargs):
    print('Initialised SGD optimizer')
    return torch.optim.SGD(parameters, lr=lr, momentum=0.9)
