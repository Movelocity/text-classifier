"""
主训练调度文件

整合数据加载、模型创建、训练和保存的完整流程。
使用SQLite数据库作为数据源，支持LoRA微调的RoBERTa模型。
"""

import os
import sys
import torch
import argparse
from datetime import datetime
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classifier.config import TrainingConfig
from classifier.sqlite_data_loader import create_data_loaders
from classifier.enhanced_trainer import EnhancedTrainer


def setup_model(config: TrainingConfig, num_labels: int, device: torch.device):
    """
    设置模型和优化器
    
    Args:
        config: 训练配置
        num_labels: 标签数量
        device: 训练设备
        
    Returns:
        model, optimizer
    """
    print(f"加载模型: {config.model_name}")
    
    # 加载tokenizer和基础模型
    tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        config.model_name, 
        num_labels=num_labels
    )
    
    # 设置为多标签分类
    model.config.problem_type = config.problem_type
    
    # 如果使用LoRA，应用LoRA配置
    if config.use_lora:
        print("应用LoRA配置...")
        lora_config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout
        )
        model = get_peft_model(model, lora_config)
        
        # 打印可训练参数
        model.print_trainable_parameters()
    
    # 移动到指定设备
    model = model.to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    return model, optimizer, tokenizer


def main(config_path: str = None, **kwargs):
    """
    主训练流程
    
    Args:
        config_path: 配置文件路径（可选）
        **kwargs: 命令行参数覆盖配置
    """
    # 加载配置
    if config_path and os.path.exists(config_path):
        print(f"从配置文件加载: {config_path}")
        config = TrainingConfig.from_yaml(config_path)
    else:
        print("使用默认配置")
        config = TrainingConfig()
    
    # 用命令行参数覆盖配置
    for key, value in kwargs.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
            print(f"配置覆盖: {key} = {value}")
    
    # 获取设备
    device = config.get_device()
    print(f"使用设备: {device}")
    
    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.save_dir, f"run_{timestamp}")
    config.save_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)
    
    print("=" * 50)
    print("开始训练流程")
    print("=" * 50)
    
    try:
        # 1. 创建数据加载器
        print("\n1. 创建数据加载器...")
        tokenizer_temp = RobertaTokenizer.from_pretrained(config.model_name)
        
        train_loader, valid_loader, test_loader, label2id, id2label = create_data_loaders(
            split_ratio=config.data_split_ratio,
            batch_size=config.batch_size,
            tokenizer=tokenizer_temp,
            device=device,
            max_length=config.max_length,
            shuffle_train=config.shuffle_train,
            random_seed=config.random_seed
        )
        
        # 更新配置中的标签数量
        config.num_labels = len(label2id)
        
        print(f"数据加载完成:")
        print(f"  - 训练样本: {len(train_loader.dataset)}")
        print(f"  - 验证样本: {len(valid_loader.dataset)}")
        print(f"  - 测试样本: {len(test_loader.dataset)}")
        print(f"  - 标签数量: {config.num_labels}")
        print(f"  - 批次大小: {config.batch_size}")
        
        # 2. 设置模型
        print("\n2. 设置模型和优化器...")
        model, optimizer, tokenizer = setup_model(config, config.num_labels, device)
        
        print(f"模型创建完成:")
        print(f"  - 模型类型: {config.model_name}")
        print(f"  - 使用LoRA: {config.use_lora}")
        print(f"  - 问题类型: {config.problem_type}")
        
        # 3. 创建训练器
        print("\n3. 创建训练器...")
        trainer = EnhancedTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=valid_loader,
            config=config,
            id2label=id2label,
            test_loader=test_loader
        )
        
        # 保存配置到运行目录
        config_save_path = os.path.join(run_dir, 'config.yaml')
        config.save_config(config_save_path)
        print(f"配置已保存到: {config_save_path}")
        
        # 4. 开始训练
        print("\n4. 开始训练...")
        best_model_path = trainer.train()
        
        print("\n" + "=" * 50)
        print("训练完成!")
        print("=" * 50)
        print(f"最佳模型保存在: {best_model_path}")
        print(f"运行目录: {run_dir}")
        
        return best_model_path, run_dir
        
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="文本分类模型训练")
    
    # 配置文件
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    # 数据相关参数
    parser.add_argument("--batch_size", type=int, help="批次大小")
    parser.add_argument("--max_length", type=int, help="最大序列长度")
    
    # 模型相关参数
    parser.add_argument("--model_name", type=str, help="预训练模型名称")
    
    # LoRA相关参数
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    parser.add_argument("--lora_r", type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout")
    
    # 训练相关参数
    parser.add_argument("--num_epochs", type=int, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--weight_decay", type=float, help="权重衰减")
    
    # 其他参数
    parser.add_argument("--save_dir", type=str, help="模型保存目录")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], help="训练设备")
    parser.add_argument("--random_seed", type=int, help="随机种子")
    
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 转换为字典，过滤掉None值
    kwargs = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
    
    # 运行主流程
    try:
        best_model_path, run_dir = main(config_path=args.config, **kwargs)
        print(f"\n✅ 训练成功完成!")
        print(f"📁 运行目录: {run_dir}")
        print(f"🎯 最佳模型: {best_model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断训练")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        sys.exit(1) 