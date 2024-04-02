import os 
import random
import chardet
import numpy as np
import typing as tp
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader



def detect_encoding(filepath):
    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read(10000))  # Read the first 10000 bytes to guess the encoding
    return result['encoding']


def load_split_data(
        input_dir:tp.Union[str, Path], 
        output_dir:tp.Union[str, Path],
        split_ratio:list=[8, 1, 1]
    )->tp.NoReturn:
    """
    - 数据文件格式: txt
    - 编码: 主流字符编码即可, utf-8, utf-16, gb2312. 切分后统一保存为utf-8
    - 文档格式:
        `整数标签\\t文本`，一行一条数据
    - 请自己决定数据集 label 的加载方式
    """    
    # =========参数验证=============
    assert len(split_ratio) == 3

    split_ratio = np.array(split_ratio)
    assert np.all(split_ratio>0)

    if not os.path.isfile(input_dir):
        raise ValueError(f'无法识别为文件: {input_dir}')
    if type(input_dir) == 'str':
        input_dir = Path(input_dir)
    if type(output_dir) == 'str':
        output_dir = Path(output_dir)
    #================================

    # 读取数据文件
    with open(input_dir, 'r', encoding=detect_encoding(input_dir)) as file:
        lines = file.readlines()

    random.shuffle(lines)  # Shuffle the lines in place

    # Calculate the split sizes
    total_lines = len(lines)
    train_ratio, eval_ratio, test_ratio = split_ratio/np.sum(split_ratio)
    train_size, eval_size = int(train_ratio * total_lines), int(eval_ratio * total_lines)
    # test_size 不用算，剩下的就是它的了
    train_lines, remaining_lines = lines[:train_size], lines[train_size:]
    eval_lines, test_lines = remaining_lines[:eval_size], remaining_lines[eval_size:]

    files_map = {
        output_dir/'train.txt': train_lines,
        output_dir/'eval.txt': eval_lines,
        output_dir/'test.txt': test_lines
    }
    for filepath, lines in files_map.items():
        # Write the splits to their respective files
        with open(filepath, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        print(f'write {len(lines)} lines to {filepath}')

def read_csv_labels(
    input_dir: tp.Union[str, Path],
    label_colname:str,
    id_colname:str
):
    df = pd.read_csv(input_dir)
    label2id = df.set_index(label_colname)[id_colname].to_dict()
    id2label = {str(i): n for n, i in label2id.items()}
    return id2label, label2id


class NormalDataset(Dataset):
    def __init__(self, file_path, tokenizer, device:torch.device, max_length=256):
        self.tokenizer = tokenizer
        self.texts = []
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.targets = []
        self.device = device
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                category, text = line.strip().split('\t')

                encoded = tokenizer.encode_plus(
                    text, 
                    truncation=True, 
                    add_special_tokens=True,
                    padding='max_length', 
                    max_length=max_length,
                    return_tensors='pt'
                )
                self.texts.append(text)
                self.input_ids.append(encoded['input_ids'].squeeze(0))
                self.token_type_ids.append(encoded['token_type_ids'].squeeze(0))
                self.attention_mask.append(encoded['attention_mask'].squeeze(0))
                self.targets.append(int(category))

        self.input_ids = torch.stack(self.input_ids)
        self.token_type_ids = torch.stack(self.token_type_ids)
        self.attention_mask = torch.stack(self.attention_mask)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx].to(self.device),
            'token_type_ids': self.token_type_ids[idx].to(self.device),
            'attention_mask': self.attention_mask[idx].to(self.device),
            'labels': self.targets[idx].to(self.device)
        }


class TextDatasetGPU(Dataset):
    def __init__(self, file_path, tokenizer, device:torch.device, max_length=256):
        self.tokenizer = tokenizer
        self.texts = []
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.targets = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                category, text = line.strip().split('\t')

                encoded = tokenizer.encode_plus(
                    text, 
                    truncation=True, 
                    add_special_tokens=True,
                    padding='max_length', 
                    max_length=max_length,
                    return_tensors='pt'
                )
                self.texts.append(text)
                self.input_ids.append(encoded['input_ids'].squeeze(0))
                self.token_type_ids.append(encoded['token_type_ids'].squeeze(0))
                self.attention_mask.append(encoded['attention_mask'].squeeze(0))
                self.targets.append(int(category))

        # Convert lists to tensors and move to GPU in advance
        self.input_ids = torch.stack(self.input_ids).to(device)
        self.token_type_ids = torch.stack(self.token_type_ids).to(device)
        self.attention_mask = torch.stack(self.attention_mask).to(device)
        self.targets = torch.tensor(self.targets, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'token_type_ids': self.token_type_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.targets[idx]
        }

