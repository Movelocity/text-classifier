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

import pandas as pd


def calculate_top_n_accuracies(preds, labels, n):
    n = min(preds.size(1), n)  # Ensure n does not exceed the number of classes
    top_n_preds = preds.topk(n, dim=1).indices  # Get the top-n predictions for each example
    labels = labels.to(preds.device)  # Move labels to the same device as preds
    # Compare with labels and calculate the number of correct predictions
    correct = top_n_preds.eq(labels.unsqueeze(1).expand_as(top_n_preds)).any(dim=1).sum().item()
    return correct


def eval_as_table(model, tokenizer, dataloader, id2label):
    top1, top3, top5 = 0, 0, 0
    total = 0

    texts = []
    preds = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels']

            # Handle the possibility that 'token_type_ids' may not be present
            token_type_ids = batch.get('token_type_ids', None)
            model_kwargs = {
                'input_ids': batch['input_ids'], 
                'attention_mask': batch['attention_mask'],
                'token_type_ids': batch['token_type_ids']
            }
            # Get the logits from the model
            outputs = model(**model_kwargs)
            probs = outputs.logits.softmax(dim=-1)
            top1 += calculate_top_n_accuracies(probs, labels, 1)
            top3 += calculate_top_n_accuracies(probs, labels, 3)
            top5 += calculate_top_n_accuracies(probs, labels, 5)
            total += labels.size(0)
            
            top_probs, top_indices = probs.topk(k=5, sorted=True)
            for indice_, label_ in zip(top_indices, batch["labels"]):
                preds.append(indice_.cpu().numpy())
                targets.append(label_.cpu().numpy())
            for ids_ in batch["input_ids"]:
                texts.append(tokenizer.decode(ids_, skip_special_tokens=True))
            
            print(f"\rEvaluated {total}/{len(dataloader.dataset)}", end="", flush=True)
    
    df = pd.DataFrame({
        "text": texts,
        "label": [id2label[t] for t in targets],
        "t1": [id2label(p[0]) for p in preds],
        "t2": [id2label(p[1]) for p in preds],
        "t3": [id2label(p[2]) for p in preds]
    })
    df.to_csv('evaluation.csv', index=False)
    return df