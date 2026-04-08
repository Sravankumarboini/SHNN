def is_fault(activations, threshold=0.01):
    return activations.mean().item() < threshold
