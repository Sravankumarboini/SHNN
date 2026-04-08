from monitoring.fault_classifier import is_fault
from healing.retraining import heal_model

class SHNN:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def forward(self, x):
        out = self.model(x)
        if is_fault(self.model.activations):
            print("⚠ Fault detected → Healing")
            heal_model(self.model, self.data_loader)
        return out
