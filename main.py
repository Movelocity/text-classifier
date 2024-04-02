import numpy as np
import torch
import matplotlib.pyplot as plt
from data_utils import TextDatasetGPU
from pathlib import Path

from classifiers import BertTokenizer, BertClassifier, load_model_eval, Trainer
from classifiers.tools import show_diff
from data_utils import DataLoader, read_csv_labels

def main():
    from transformers import AutoConfig
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    bert_base_dir = Path('/kaggle/input/code-test/bert')
    model = load_model_eval(bert_base_dir, device=device)

    train_dataset = TextDatasetGPU('train.txt', model.tokenizer)
    eval_dataset = TextDatasetGPU('eval.txt', model.tokenizer)
    test_dataset = TextDatasetGPU('test.txt', model.tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=24, shuffle=False)


    print(len(train_dataset.input_ids))

    id2label, label2id = read_csv_labels()
    config = AutoConfig.from_pretrained(
        bert_base_dir, 
        label2id=label2id,
        id2label=id2label,
        hidden_dropout_prob=0.3,
        num_labels=len(label2id.keys()),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-8)

    trainer = Trainer(model, optimizer, train_loader, eval_loader, device)
    trainer.train(5)

    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=True)
    for batch in test_loader:
        with torch.no_grad():
            probs = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask'], 
                token_type_ids=batch['token_type_ids']
            )
        break
    show_diff(
        probs.detach().cpu().numpy(), 
        batch['labels'].cpu().numpy()
    )