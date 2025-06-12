import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
# from plot_utils import multiline_plot
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import shared

# print(f"detect classes {shared.num_classes}")

class AUCMetric:
    def __init__(self):
        self.predictions = []
        self.references = []

    def reset(self):
        self.predictions = []
        self.references = []

    def add_batch(self, predictions, references):
        # Ensure predictions are softmax probabilities if needed
        self.predictions.append(predictions.detach().cpu().numpy())
        self.references.append(references.detach().cpu().numpy())

    def compute(self):
        predictions_np = np.concatenate(self.predictions)
        references_np = np.concatenate(self.references)
        
        auc = roc_auc_score(references_np, predictions_np, multi_class='ovr', average='micro')
        self.reset()
        return auc

# class AccuracyMetric:
#     def __init__(self,):
#         self.reset()

#     def reset(self):
#         self.matches = 0
#         self.total_samples = 0

#     def add_batch(self, predictions, references):
#         self.matches += torch.eq(predictions, references).sum().item()
#         self.total_samples += references.size(0)
    
#     def compute(self):
#         result = self.matches/self.total_samples
#         self.reset()
#         return result

class Trainer:
    def __init__(self, model, optimizer, train_loader, eval_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.gradient_accumulation_steps = 4
        self.device = device
        self.num_epochs = 10

        self.metric = AUCMetric()
        self.reset_logs()

    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.1)

    def train_epoch(self):
        self.model.train()
        epoch_loss = []
        for step, batch in enumerate(tqdm(self.train_loader, desc='train')):
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.clip_grad_norm()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            epoch_loss.append(loss.item())
        return np.mean(epoch_loss)
    
    def eval_epoch(self):
        self.model.eval()
        for step, batch in enumerate(self.eval_loader):
            with torch.no_grad():
                model_kwargs = {
                    'input_ids': batch['input_ids'], 
                    'attention_mask': batch['attention_mask'],
                    # 'token_type_ids': batch['token_type_ids']
                }
                outputs = self.model(**model_kwargs)
            self.metric.add_batch(
                predictions=outputs.logits.sigmoid(),
                references=batch["labels"],
            )
        return self.metric.compute()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch()
            eval_score = self.eval_epoch()
            self.loss_trace.append(epoch_loss)
            self.acc_trace.append(eval_score)
            print(f"epoch {epoch}: auc: {eval_score}, loss: {epoch_loss}")
    
    def plot_history(self):
        xs = range(len(self.loss_trace))
        # Create a figure with two subplots in a row
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss on the first subplot
        ax1.plot(xs, self.loss_trace, color='blue', label='Loss')
        ax1.set_title('Loss over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot AUC on the second subplot
        ax2.plot(xs, self.acc_trace, color='orange', label='AUC')
        ax2.set_title('AUC over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.legend()
        ax2.grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def reset_logs(self):
        self.step = 0
        self.loss_trace = []
        self.xs = []
        self.acc_trace = []
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0.06 * (len(self.train_loader) * self.num_epochs),
            num_training_steps=(len(self.train_loader) * self.num_epochs),
        )
