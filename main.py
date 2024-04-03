import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from data_utils import load_split_data, read_csv_labels
from data_utils import TextDatasetGPU, DataLoader

# load_split_data(Path('/kaggle/input/ebay-intent-36/ebay36.txt'), Path('.'))

id2label, label2id = read_csv_labels(
    '/kaggle/input/ebay-intent-36/class_labels.csv',
    label_colname='chinese',
    id_colname='id'
)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



from models import cls_model_with_adapter
from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification
base_model_dir = '/kaggle/input/code-test/bert'
config = AutoConfig.from_pretrained(
    base_model_dir, 
    problem_type="single_label_classification",
    hidden_dropout_prob=0.3,
    num_labels=36,  # Adapter 中包含 bert.classifier 的权重，如果形状不对就会报错
)
base_model = BertForSequenceClassification.from_pretrained(
    base_model_dir, config=config, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

train_dataset = TextDatasetGPU('train.txt', tokenizer, device)
eval_dataset = TextDatasetGPU('eval.txt', tokenizer, device)
test_dataset = TextDatasetGPU('test.txt', tokenizer, device)

train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=24, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=48, shuffle=True)



from peft import LoraModel, LoraConfig, get_peft_model
lora_config = LoraConfig(
    task_type="SEQ_CLS", inference_mode=False, r=24, lora_alpha=24, lora_dropout=0.1)
adapted_model = get_peft_model(base_model, lora_config).to(device)
optimizer = torch.optim.AdamW(adapted_model.parameters(), lr=3e-4, eps=1e-8)

adapted_model.print_trainable_parameters()


from eval_utils import AccuracyMetric
from plot_utils import multiline_plot
from trainer import Trainer

trainer = Trainer(adapted_model, optimizer, train_loader, eval_loader, device)

trainer.train(5)
trainer.plot_history()