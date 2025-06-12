"""
SQLite数据加载器

这个模块实现从SQLite数据库加载训练数据的功能，用于替代原有的文件加载方式。
包含数据集类和相关的数据处理功能。
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from sqlalchemy.orm import Session
from sqlalchemy import func

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlite_as_dataset.models import AnnotationData, SessionLocal
from sqlite_as_dataset.services import AnnotationService


def parse_labels(labels_str: Optional[str]) -> List[str]:
    """
    解析标签字符串为标签列表。
    
    Args:
        labels_str: 逗号分隔的标签字符串
        
    Returns:
        标签列表
    """
    if not labels_str:
        return []
    return [label.strip() for label in labels_str.split(',') if label.strip()]


class SQLiteDatasetLoader:
    """从SQLite数据库加载数据的类"""
    
    def __init__(self, db_session: Session = None):
        """
        初始化SQLite数据加载器
        
        Args:
            db_session: 数据库会话，如果为None则创建新会话
        """
        self.db = db_session or SessionLocal()
        self.annotation_service = AnnotationService(self.db)
    
    def get_all_labels(self) -> Dict[str, int]:
        """
        获取所有唯一标签并创建标签到ID的映射
        
        Returns:
            标签名到ID的映射字典
        """
        # 查询所有带标签的记录
        labeled_data = self.db.query(AnnotationData).filter(
            AnnotationData.labels.is_not(None),
            AnnotationData.labels != ''
        ).all()
        
        # 收集所有唯一标签
        all_labels = set()
        for item in labeled_data:
            labels = parse_labels(item.labels)
            all_labels.update(labels)
        
        # 创建标签到ID的映射（按字母顺序排序）
        sorted_labels = sorted(all_labels)
        label2id = {label: idx for idx, label in enumerate(sorted_labels)}
        
        return label2id
    
    def load_data_split(self, split_ratio: List[float] = [0.8, 0.1, 0.1], 
                       shuffle: bool = True, random_seed: int = 42) -> Tuple[List, List, List]:
        """
        从SQLite加载数据并按比例分割为训练、验证、测试集
        
        Args:
            split_ratio: 分割比例 [train, valid, test]
            shuffle: 是否随机打乱数据
            random_seed: 随机种子
            
        Returns:
            (train_data, valid_data, test_data) 三个数据集的列表
        """
        # 验证分割比例
        assert len(split_ratio) == 3, "split_ratio必须包含3个值"
        assert abs(sum(split_ratio) - 1.0) < 1e-6, "split_ratio的和必须为1.0"
        
        # 查询所有带标签的数据
        labeled_data = self.db.query(AnnotationData).filter(
            AnnotationData.labels.is_not(None),
            AnnotationData.labels != ''
        ).all()
        
        print(f"从数据库加载了 {len(labeled_data)} 条有标签的数据")
        
        # 转换为列表格式
        data_list = []
        for item in labeled_data:
            data_list.append({
                'id': item.id,
                'text': item.text,
                'labels': item.labels
            })
        
        # 随机打乱数据
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(data_list)
        
        # 计算分割点
        total_size = len(data_list)
        train_size = int(total_size * split_ratio[0])
        valid_size = int(total_size * split_ratio[1])
        
        # 分割数据
        train_data = data_list[:train_size]
        valid_data = data_list[train_size:train_size + valid_size]
        test_data = data_list[train_size + valid_size:]
        
        print(f"数据分割: 训练集={len(train_data)}, 验证集={len(valid_data)}, 测试集={len(test_data)}")
        
        return train_data, valid_data, test_data
    
    def close(self):
        """关闭数据库连接"""
        if self.db:
            self.db.close()


class SQLiteTextDataset(Dataset):
    """
    基于SQLite数据的PyTorch Dataset
    支持多标签分类
    """
    
    def __init__(self, data: List[Dict], tokenizer, device: torch.device, 
                 label2id: Dict[str, int], max_length: int = 256):
        """
        初始化数据集
        
        Args:
            data: 数据列表，每个元素包含id, text, labels
            tokenizer: 分词器
            device: 设备
            label2id: 标签到ID的映射
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.device = device
        self.label2id = label2id
        self.num_classes = len(label2id)
        self.max_length = max_length
        
        # 预处理数据
        self._preprocess_data()
    
    def _preprocess_data(self):
        """预处理数据：分词和编码"""
        self.texts = []
        self.input_ids = []
        self.attention_mask = []
        self.targets = []
        
        skipped = 0
        multi_label_count = 0
        
        for item in tqdm(self.data, desc="预处理数据"):
            text = item['text']
            labels_str = item['labels']
            
            # 解析标签
            labels = parse_labels(labels_str)
            if not labels:
                skipped += 1
                continue
            
            # 统计多标签数据
            if len(labels) > 1:
                multi_label_count += 1
            
            # 编码文本
            encoded = self.tokenizer.encode_plus(
                text,
                truncation=True,
                add_special_tokens=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # 创建多热编码标签
            label_ids = [self.label2id[label] for label in labels if label in self.label2id]
            
            self.texts.append(text)
            self.input_ids.append(encoded['input_ids'].squeeze(0))
            self.attention_mask.append(encoded['attention_mask'].squeeze(0))
            self.targets.append(label_ids)
        
        # 转换为张量并移到指定设备
        self.input_ids = torch.stack(self.input_ids).to(self.device)
        self.attention_mask = torch.stack(self.attention_mask).to(self.device)
        
        print(f"跳过的记录: {skipped}")
        print(f"多标签记录: {multi_label_count}")
        print(f"有效记录: {len(self.texts)}")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        # 创建多热编码
        multi_hot = torch.zeros(self.num_classes, dtype=torch.float32).to(self.device)
        for label_id in self.targets[idx]:
            multi_hot[label_id] = 1.0
        
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': multi_hot
        }


def create_data_loaders(split_ratio: List[float] = [0.8, 0.1, 0.1],
                       batch_size: int = 16,
                       tokenizer=None,
                       device: torch.device = None,
                       max_length: int = 256,
                       shuffle_train: bool = True,
                       random_seed: int = 42) -> Tuple:
    """
    创建训练、验证、测试数据加载器
    
    Args:
        split_ratio: 数据分割比例
        batch_size: 批次大小
        tokenizer: 分词器
        device: 设备
        max_length: 最大序列长度
        shuffle_train: 是否打乱训练数据
        random_seed: 随机种子
        
    Returns:
        (train_loader, valid_loader, test_loader, label2id, id2label)
    """
    # 创建数据加载器
    loader = SQLiteDatasetLoader()
    
    try:
        # 获取标签映射
        label2id = loader.get_all_labels()
        id2label = {idx: label for label, idx in label2id.items()}
        
        print(f"发现 {len(label2id)} 个唯一标签")
        
        # 加载和分割数据
        train_data, valid_data, test_data = loader.load_data_split(
            split_ratio=split_ratio,
            shuffle=True,
            random_seed=random_seed
        )
        
        # 创建数据集
        train_dataset = SQLiteTextDataset(train_data, tokenizer, device, label2id, max_length)
        valid_dataset = SQLiteTextDataset(valid_data, tokenizer, device, label2id, max_length)
        test_dataset = SQLiteTextDataset(test_data, tokenizer, device, label2id, max_length)
        
        # 创建数据加载器
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle_train
        )
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        return train_loader, valid_loader, test_loader, label2id, id2label
        
    finally:
        loader.close() 