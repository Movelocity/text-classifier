"""
增强的训练器模块

提供更完善的训练功能，包括：
- 训练过程跟踪
- 模型保存管理
- 详细的日志记录
- 早停机制
"""

import os
import json
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report
from transformers import get_linear_schedule_with_warmup

from .config import TrainingConfig


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_auc': [],
            'learning_rate': [],
            'epoch': [],
            'step': []
        }
        
        # 创建日志文件
        self.log_file = os.path.join(log_dir, 'training.log')
        self.metrics_file = os.path.join(log_dir, 'metrics.json')
    
    def log(self, message: str, level: str = "INFO"):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def log_metrics(self, metrics: Dict, step: int, epoch: int):
        """记录训练指标"""
        # 更新历史记录
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        self.metrics_history['step'].append(step)
        self.metrics_history['epoch'].append(epoch)
        
        # 保存到文件
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """绘制训练指标图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练损失
        if self.metrics_history['train_loss']:
            axes[0, 0].plot(self.metrics_history['epoch'], self.metrics_history['train_loss'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # 验证AUC
        if self.metrics_history['eval_auc']:
            axes[0, 1].plot(self.metrics_history['epoch'], self.metrics_history['eval_auc'])
            axes[0, 1].set_title('Validation AUC')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('AUC')
            axes[0, 1].grid(True)
        
        # 学习率
        if self.metrics_history['learning_rate']:
            axes[1, 0].plot(self.metrics_history['step'], self.metrics_history['learning_rate'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].grid(True)
        
        # 验证损失
        if self.metrics_history['eval_loss']:
            axes[1, 1].plot(self.metrics_history['epoch'], self.metrics_history['eval_loss'])
            axes[1, 1].set_title('Validation Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class AUCMetric:
    """AUC指标计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.predictions = []
        self.references = []
    
    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor):
        """添加一个批次的预测结果"""
        # 确保预测值是概率形式
        if predictions.dim() > 1:
            predictions = torch.sigmoid(predictions)
        
        self.predictions.append(predictions.detach().cpu().numpy())
        self.references.append(references.detach().cpu().numpy())
    
    def compute(self) -> float:
        """计算AUC分数"""
        if not self.predictions:
            return 0.0
        
        predictions_np = np.concatenate(self.predictions, axis=0)
        references_np = np.concatenate(self.references, axis=0)
        
        try:
            # 使用micro平均计算多标签AUC
            auc = roc_auc_score(references_np, predictions_np, 
                              multi_class='ovr', average='micro')
            return auc
        except Exception as e:
            print(f"AUC计算错误: {e}")
            return 0.0
        finally:
            self.reset()


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, 
                 mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        
    def __call__(self, score: float) -> bool:
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


class EnhancedTrainer:
    """增强的训练器"""
    
    def __init__(self, model, optimizer, train_loader, eval_loader, 
                 config: TrainingConfig, id2label: Dict[int, str],
                 test_loader=None):
        """
        初始化增强训练器
        
        Args:
            model: 模型
            optimizer: 优化器
            train_loader: 训练数据加载器
            eval_loader: 验证数据加载器
            config: 训练配置
            id2label: ID到标签的映射
            test_loader: 测试数据加载器（可选）
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.config = config
        self.id2label = id2label
        self.device = config.get_device()
        
        # 创建学习率调度器
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # 创建指标和日志
        self.auc_metric = AUCMetric()
        self.logger = TrainingLogger(os.path.join(config.save_dir, 'logs'))
        
        # 早停机制
        self.early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='max')
        
        # 训练状态
        self.global_step = 0
        self.best_auc = 0.0
        self.best_model_path = None
        
        self.logger.log(f"训练器初始化完成，设备: {self.device}")
        self.logger.log(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")
    
    def save_model(self, save_path: str, metrics: Dict, is_best: bool = False):
        """保存模型和相关信息"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(save_path)
        
        # 保存配置
        config_path = os.path.join(save_path, 'training_config.yaml')
        self.config.save_config(config_path)
        
        # 保存标签映射
        label_config = {
            'id2label': self.id2label,
            'label2id': {v: k for k, v in self.id2label.items()}
        }
        
        with open(os.path.join(save_path, 'label_config.json'), 'w', encoding='utf-8') as f:
            json.dump(label_config, f, indent=2, ensure_ascii=False)
        
        # 保存训练指标
        with open(os.path.join(save_path, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        if is_best:
            self.best_model_path = save_path
            self.logger.log(f"保存最佳模型到: {save_path}")
        else:
            self.logger.log(f"保存模型到: {save_path}")
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="训练")
        
        for batch in progress_bar:
            # 前向传播
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # 梯度累积
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 梯度更新
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             self.config.max_grad_norm)
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # 记录学习率
                current_lr = self.lr_scheduler.get_last_lr()[0]
                
                # 定期记录指标
                if self.global_step % self.config.logging_steps == 0:
                    metrics = {
                        'train_loss': loss.item() * self.config.gradient_accumulation_steps,
                        'learning_rate': current_lr
                    }
                    self.logger.log_metrics(metrics, self.global_step, -1)
            
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
            })
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        self.auc_metric.reset()
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="评估"):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # 计算损失
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(outputs.logits, batch['labels'])
                
                total_loss += loss.item()
                num_batches += 1
                
                # 收集预测结果
                self.auc_metric.add_batch(outputs.logits, batch['labels'])
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        auc_score = self.auc_metric.compute()
        
        return avg_loss, auc_score
    
    def train(self) -> str:
        """执行完整的训练流程"""
        self.logger.log("开始训练...")
        
        for epoch in range(self.config.num_epochs):
            self.logger.log(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            
            # 评估
            eval_loss, eval_auc = self.evaluate()
            
            # 记录指标
            metrics = {
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'eval_auc': eval_auc,
                'learning_rate': self.lr_scheduler.get_last_lr()[0]
            }
            
            self.logger.log_metrics(metrics, self.global_step, epoch)
            
            self.logger.log(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Eval Loss: {eval_loss:.4f}, "
                f"Eval AUC: {eval_auc:.4f}"
            )
            
            # 保存检查点
            is_best = eval_auc > self.best_auc
            if is_best:
                self.best_auc = eval_auc
            
            if (epoch + 1) % (self.config.save_steps // len(self.train_loader)) == 0 or is_best:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = os.path.join(
                    self.config.save_dir, 
                    f"checkpoint_epoch_{epoch + 1}_{timestamp}"
                )
                self.save_model(save_dir, metrics, is_best)
            
            # 早停检查
            if self.early_stopping(eval_auc):
                self.logger.log(f"早停触发，在epoch {epoch + 1}停止训练")
                break
        
        # 保存最终模型
        final_save_dir = os.path.join(
            self.config.save_dir, 
            f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        final_metrics = {
            'final_train_loss': train_loss,
            'final_eval_loss': eval_loss,
            'final_eval_auc': eval_auc,
            'best_auc': self.best_auc,
            'total_epochs': epoch + 1
        }
        self.save_model(final_save_dir, final_metrics)
        
        # 绘制训练曲线
        plot_path = os.path.join(self.config.save_dir, 'training_curves.png')
        self.logger.plot_metrics(plot_path)
        
        self.logger.log("训练完成!")
        self.logger.log(f"最佳AUC: {self.best_auc:.4f}")
        self.logger.log(f"最佳模型路径: {self.best_model_path}")
        
        return self.best_model_path or final_save_dir 