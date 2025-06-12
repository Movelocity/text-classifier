"""
模型加载和评估模块

提供训练完成后快速加载模型进行推理和评估的功能，特别适合在Jupyter Notebook中使用。
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("⚠️  PEFT未安装，无法加载LoRA模型")

from sklearn.metrics import roc_auc_score


class ModelEvaluator:
    """
    模型评估器
    
    支持加载训练好的模型，进行单条预测、批量预测和完整评估
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        初始化模型评估器
        
        Args:
            model_path: 模型保存路径
            device: 设备 ("auto", "cpu", "cuda")
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        
        # 加载配置和标签映射
        self._load_configs()
        
        # 加载模型和分词器
        self._load_model()
        
        print(f"✅ 模型评估器初始化完成")
        print(f"🤖 模型路径: {model_path}")
        print(f"🖥️  设备: {self.device}")
        print(f"🏷️  标签数量: {len(self.id2label)}")
    
    def _get_device(self, device: str) -> torch.device:
        """获取设备"""
        if device == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == "cuda":
            if not torch.cuda.is_available():
                print("⚠️  CUDA不可用，切换到CPU")
                return torch.device('cpu')
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _load_configs(self):
        """加载配置文件"""
        # 加载标签配置
        label_config_path = os.path.join(self.model_path, 'label_config.json')
        if os.path.exists(label_config_path):
            with open(label_config_path, 'r', encoding='utf-8') as f:
                label_config = json.load(f)
                self.id2label = {int(k): v for k, v in label_config['id2label'].items()}
                self.label2id = label_config['label2id']
        else:
            raise FileNotFoundError(f"标签配置文件不存在: {label_config_path}")
        
        # 加载训练配置（可选）
        train_config_path = os.path.join(self.model_path, 'training_config.yaml')
        self.train_config = None
        if os.path.exists(train_config_path):
            try:
                import yaml
                with open(train_config_path, 'r', encoding='utf-8') as f:
                    self.train_config = yaml.safe_load(f)
            except:
                print("⚠️  无法加载训练配置文件")
        
        # 加载训练指标（可选）
        metrics_path = os.path.join(self.model_path, 'metrics.json')
        self.train_metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                self.train_metrics = json.load(f)
    
    def _load_model(self):
        """加载模型和分词器"""
        print("🔄 加载模型...")
        
        # 确定基础模型名称
        config_path = os.path.join(self.model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
                base_model_name = model_config.get('_name_or_path', 'roberta-base')
        else:
            base_model_name = 'roberta-base'
        
        # 加载分词器
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model_name)
        
        # 检查是否是LoRA模型
        adapter_config_path = os.path.join(self.model_path, 'adapter_config.json')
        if os.path.exists(adapter_config_path) and HAS_PEFT:
            # LoRA模型
            print("🔧 检测到LoRA模型，加载基础模型和适配器...")
            base_model = RobertaForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=len(self.id2label)
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            # 完整模型
            print("🤖 加载完整模型...")
            self.model = RobertaForSequenceClassification.from_pretrained(self.model_path)
        
        # 设置为评估模式并移到指定设备
        self.model.eval()
        self.model.to(self.device)
        
        print("✅ 模型加载完成")
    
    def predict_single(self, text: str, threshold: float = 0.5, 
                      return_probabilities: bool = False) -> Union[List[str], Dict[str, Any]]:
        """
        单条文本预测
        
        Args:
            text: 输入文本
            threshold: 预测阈值
            return_probabilities: 是否返回概率分数
            
        Returns:
            预测的标签列表或详细结果字典
        """
        # 编码文本
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        # 移到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # 获取预测标签
        predicted_labels = []
        label_scores = {}
        
        for i, prob in enumerate(probabilities):
            label = self.id2label[i]
            label_scores[label] = float(prob)
            if prob > threshold:
                predicted_labels.append(label)
        
        if return_probabilities:
            return {
                'text': text,
                'predicted_labels': predicted_labels,
                'all_scores': label_scores,
                'threshold': threshold
            }
        else:
            return predicted_labels
    
    def predict_batch(self, texts: List[str], threshold: float = 0.5, 
                     batch_size: int = 32, show_progress: bool = True) -> List[List[str]]:
        """
        批量预测
        
        Args:
            texts: 文本列表
            threshold: 预测阈值
            batch_size: 批次大小
            show_progress: 是否显示进度条
            
        Returns:
            每个文本的预测标签列表
        """
        results = []
        
        # 分批处理
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="批量预测")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                pred_labels = self.predict_single(text, threshold, return_probabilities=False)
                batch_results.append(pred_labels)
            
            results.extend(batch_results)
        
        return results
    
    def print_model_info(self):
        """打印模型信息"""
        print("=" * 50)
        print("🤖 模型信息")
        print("=" * 50)
        print(f"📁 模型路径: {self.model_path}")
        print(f"🖥️  设备: {self.device}")
        print(f"🏷️  标签数量: {len(self.id2label)}")
        
        # 模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 总参数量: {total_params:,}")
        print(f"🔧 可训练参数: {trainable_params:,}")
        
        if self.train_metrics:
            print(f"\n📈 训练指标:")
            for key, value in self.train_metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
        
        print(f"\n🏷️  所有标签: {', '.join(self.id2label.values())}")


def load_model(model_path: str, device: str = "auto") -> ModelEvaluator:
    """
    快速加载模型的便捷函数
    
    Args:
        model_path: 模型路径
        device: 设备
        
    Returns:
        ModelEvaluator实例
    """
    return ModelEvaluator(model_path, device) 