"""
训练配置文件

包含模型训练的所有参数配置
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingConfig:
    """训练配置类"""
    
    # 数据相关参数
    data_split_ratio: List[float] = None  # 数据分割比例 [train, valid, test]
    batch_size: int = 16
    max_length: int = 256
    shuffle_train: bool = True
    random_seed: int = 42
    max_samples: Optional[int] = None  # 最大样本数量限制（用于小规模测试）
    
    # 模型相关参数
    model_name: str = "roberta-base"
    num_labels: Optional[int] = None  # 会在运行时动态设置
    problem_type: str = "multi_label_classification"
    
    # LoRA相关参数
    use_lora: bool = True
    lora_r: int = 48
    lora_alpha: int = 24
    lora_dropout: float = 0.3
    
    # 训练相关参数
    num_epochs: int = 12
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.1
    
    # 保存和日志相关参数
    save_dir: str = "models"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # 设备相关参数
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # 其他参数
    dataloader_num_workers: int = 0
    
    def __post_init__(self):
        """初始化后处理"""
        if self.data_split_ratio is None:
            self.data_split_ratio = [0.8, 0.1, 0.1]
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
    
    def get_device(self):
        """获取训练设备"""
        import torch
        
        if self.device == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif self.device == "cuda":
            if not torch.cuda.is_available():
                print("警告: CUDA不可用，切换到CPU")
                return torch.device('cpu')
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def save_config(self, file_path: str):
        """保存配置到文件"""
        import yaml
        
        config_dict = {
            'data_split_ratio': self.data_split_ratio,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'shuffle_train': self.shuffle_train,
            'random_seed': self.random_seed,
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'problem_type': self.problem_type,
            'use_lora': self.use_lora,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_ratio': self.warmup_ratio,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_grad_norm': self.max_grad_norm,
            'save_dir': self.save_dir,
            'save_steps': self.save_steps,
            'eval_steps': self.eval_steps,
            'logging_steps': self.logging_steps,
            'save_total_limit': self.save_total_limit,
            'device': self.device,
            'dataloader_num_workers': self.dataloader_num_workers,
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def from_yaml(cls, file_path: str):
        """从YAML文件加载配置"""
        import yaml
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict) 