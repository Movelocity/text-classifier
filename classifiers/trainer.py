import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
import pandas as pd 
from tools import plot_training_history

def calculate_top_n_accuracies(preds, labels, n):
    n = min(preds.size(1), n)  # Ensure n does not exceed the number of classes
    top_n_preds = preds.topk(n, dim=1).indices  # Get the top-n predictions for each example
    labels = labels.to(preds.device)  # Move labels to the same device as preds
    # Compare with labels and calculate the number of correct predictions
    correct = top_n_preds.eq(labels.unsqueeze(1).expand_as(top_n_preds)).any(dim=1).sum().item()
    return correct

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Trainer:
    def __init__(self, model, optimizer, train_loader, eval_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.gradient_accumulation_steps = 4
        self.reset_logs()

    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.1)

    def train_epoch(self):
        self.model.train()
        cal_loss = []
        for batch in tqdm(self.train_loader):
            #with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, probs = self.model(**batch)
            loss /= self.gradient_accumulation_steps
            loss.backward()
            #scaler.scale(loss).backward()
            
            self.step += 1
            if self.step % self.gradient_accumulation_steps == 0:
                #scaler.unscale_(optimizer)
                self.clip_grad_norm()
                #scaler.step(optimizer)
                self.optimizer.step()
                self.scheduler.step()
                #scaler.update()
                self.optimizer.zero_grad()
            cal_loss.append(loss.item())
        return np.mean(cal_loss)
    
    def evaluate(self):
        top1, top3, top5 = 0, 0, 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for batch in self.eval_loader:
                labels = batch['labels']

                # Handle the possibility that 'token_type_ids' may not be present
                token_type_ids = batch.get('token_type_ids', None)
                model_kwargs = {
                    'input_ids': batch['input_ids'], 
                    'attention_mask': batch['attention_mask'],
                    'token_type_ids': batch['token_type_ids']
                }

                # Get the logits from the model
                logits = self.model(**model_kwargs)
                probs = logits.softmax(dim=-1)
                top1 += calculate_top_n_accuracies(probs, labels, 1)
                top3 += calculate_top_n_accuracies(probs, labels, 3)
                top5 += calculate_top_n_accuracies(probs, labels, 5)
                total += labels.size(0)
        # Calculate the final accuracies
        topk = {
            "top1": top1 / total,
            "top3": top3 / total,
            "top5": top5 / total
        }
        return topk

    

    def train(self, num_epochs):
        # Train the model
        # scaler = torch.cuda.amp.GradScaler()
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=20,
            num_training_steps=len(self.train_loader) * num_epochs / self.gradient_accumulation_steps
        )
        for epoch in num_epochs:
            mean_loss = self.train_epoch()
            topk = self.evaluate()
            print(f"t1: {topk['top1']:.3f}, t3: {topk['top3']:.3f}, t5: {topk['top5']:.3f}, loss: {mean_loss:.3f} [{epoch}]")

            self.xs.append(self.step)

            self.losses.append(mean_loss)
            self.t1s.append(topk['top1'])
            self.t5s.append(topk['top5'])
            
        plot_training_history(
            (5,4), xs=self.xs, 
            data=[self.losses, self.t1s, self.t5s], 
            labels=['loss', 'top1', 'top5'], 
            colors=['blue', 'green', 'red']
        )

    def reset_logs(self):
        self.step = 0
        self.losses = []
        self.xs = []
        self.t1s, self.t3s, self.t5s = [], [], []