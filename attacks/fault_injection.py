import torch

def inject_fault(model):
    model.conv.weight.data *= 0.4