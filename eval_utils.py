import torch

class AccuracyMetric:
    def __init__(self,):
        self.reset()

    def reset(self):
        self.matches = 0
        self.total_samples = 0

    def add_batch(self, predictions, references):
        self.matches += torch.eq(predictions, references).sum().item()
        self.total_samples += references.size(0)
    
    def compute(self):
        result = {
            'accuracy': self.matches/self.total_samples,
        }
        self.reset()
        return result

accuracy_metric = AccuracyMetric()