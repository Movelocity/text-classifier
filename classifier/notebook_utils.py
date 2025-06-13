"""
Notebook友好的训练配置和工具

这个模块提供了在Jupyter Notebook中便于使用的训练配置和启动方式。
"""

import os
import sys
import torch
import shutil
import yaml
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import TrainingConfig
from .sqlite_data_loader import create_data_loaders
from .enhanced_trainer import EnhancedTrainer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model


class NotebookTrainingConfig:
    """
    Notebook友好的训练配置类
    
    提供简单的方法来配置训练参数，特别适合在Jupyter Notebook中使用
    """
    
    def __init__(self):
        """初始化默认配置"""
        self.config = TrainingConfig()
        self._display_config()
    
    def _display_config(self):
        """显示当前配置"""
        print("🔧 当前训练配置:")
        print(f"  📊 数据分割: {self.config.data_split_ratio}")
        if self.config.max_samples is not None:
            print(f"  📝 数据量限制: {self.config.max_samples:,} 条")
        else:
            print(f"  📝 数据量限制: 无限制")
        print(f"  📦 批次大小: {self.config.batch_size}")
        print(f"  📏 最大长度: {self.config.max_length}")
        print(f"  🤖 模型名称: {self.config.model_name}")
        print(f"  🔧 使用LoRA: {self.config.use_lora}")
        print(f"  🔄 训练轮数: {self.config.num_epochs}")
        print(f"  📈 学习率: {self.config.learning_rate}")
        print(f"  💾 保存目录: {self.config.save_dir}")
    
    def quick_setup(
        self, 
        batch_size: int = 16,
        num_epochs: int = 12,
        learning_rate: float = 2e-4,
        max_length: int = 256,
        use_lora: bool = True
    ):
        """
        快速设置常用参数
        
        Args:
            batch_size: 批次大小
            num_epochs: 训练轮数
            learning_rate: 学习率
            max_length: 最大序列长度
            use_lora: 是否使用LoRA
        """
        self.config.batch_size = batch_size
        self.config.num_epochs = num_epochs
        self.config.learning_rate = learning_rate
        self.config.max_length = max_length
        self.config.use_lora = use_lora
        
        print("✅ 配置已更新")
        self._display_config()
        return self
    
    def set_model(self, model_name: str):
        """设置模型名称"""
        self.config.model_name = model_name
        print(f"✅ 模型设置为: {model_name}")
        return self
    
    def set_lora_config(self, r: int = 48, alpha: int = 24, dropout: float = 0.3):
        """设置LoRA参数"""
        self.config.lora_r = r
        self.config.lora_alpha = alpha
        self.config.lora_dropout = dropout
        print(f"✅ LoRA配置: r={r}, alpha={alpha}, dropout={dropout}")
        return self
    
    def set_data_split(self, train: float = 0.8, valid: float = 0.1, test: float = 0.1):
        """设置数据分割比例"""
        assert abs(train + valid + test - 1.0) < 1e-6, "分割比例之和必须为1.0"
        self.config.data_split_ratio = [train, valid, test]
        print(f"✅ 数据分割: 训练={train}, 验证={valid}, 测试={test}")
        return self
    
    def set_max_samples(self, max_samples: Optional[int]):
        """
        设置最大样本数量限制
        
        Args:
            max_samples: 最大样本数量，None表示不限制
        """
        self.config.max_samples = max_samples
        if max_samples is None:
            print("✅ 数据量限制已移除")
        else:
            print(f"✅ 数据量限制设为: {max_samples:,} 条")
        return self
    
    def set_save_dir(self, save_dir: str):
        """设置保存目录"""
        self.config.save_dir = save_dir
        print(f"✅ 保存目录: {save_dir}")
        return self
    
    def cpu_mode(self):
        """设置为CPU模式（适合小规模测试）"""
        self.config.device = "cpu"
        self.config.batch_size = min(self.config.batch_size, 8)  # 限制批次大小
        print("🖥️  已切换到CPU模式，批次大小已调整")
        return self
    
    def gpu_mode(self):
        """设置为GPU模式"""
        self.config.device = "cuda"
        print("🚀 已切换到GPU模式")
        return self
    
    def auto_mode(self):
        """自动选择设备"""
        self.config.device = "auto"
        print("🤖 已设置为自动选择设备")
        return self
    
    def small_test(self):
        """小规模测试配置（快速验证）"""
        self.config.batch_size = 4
        self.config.num_epochs = 2
        self.config.max_length = 128
        self.config.save_steps = 50
        self.config.eval_steps = 50
        self.config.logging_steps = 10
        self.config.max_samples = 10000  # 限制数据量为1万条
        print("🧪 已设置为小规模测试模式（数据量限制：10,000条）")
        self._display_config()
        return self
    
    def production_setup(self):
        """生产环境配置"""
        self.config.batch_size = 32
        self.config.num_epochs = 20
        self.config.max_length = 512
        self.config.learning_rate = 1e-4
        self.config.warmup_ratio = 0.1
        print("🏭 已设置为生产环境模式")
        self._display_config()
        return self
    
    def start_training(self, verbose: bool = True, package_model_flag: bool = True):
        """启动训练的便捷方法"""
        return start_training(self, verbose=verbose, package_model_flag=package_model_flag)


def setup_model(config: TrainingConfig, num_labels: int, device: torch.device):
    """
    设置模型和优化器
    
    Args:
        config: 训练配置
        num_labels: 标签数量
        device: 训练设备
        
    Returns:
        model, optimizer, tokenizer
    """
    print(f"🤖 加载模型: {config.model_name}")
    
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
        print("🔧 应用LoRA配置...")
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


def create_label_config(id2label: dict[int, str], save_dir: str) -> str:
    """
    创建标签配置文件
    
    Args:
        id2label: 标签ID到标签名的映射
        save_dir: 保存目录
        
    Returns:
        标签配置文件路径
    """
    label_config = {
        "id2label": {k: v for k, v in id2label.items()},
        "label2id": {v: k for k, v in id2label.items()}
    }
    
    label_config_path = os.path.join(save_dir, "label_config.yaml")
    with open(label_config_path, "w", encoding="utf-8") as f:
        yaml.dump(label_config, f, default_flow_style=False, allow_unicode=True)
    
    return label_config_path


def package_model(best_model_path: str, run_dir: str, id2label: dict, config: object) -> str:
    """
    打包训练好的模型为zip文件，便于下载
    
    Args:
        best_model_path: 最佳模型路径
        run_dir: 运行目录
        id2label: 标签映射
        config: 训练配置
        
    Returns:
        打包后的zip文件路径
    """
    print("\n📦 5. 打包模型...")
    
    # 创建打包目录
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    if config.use_lora:
        package_name = f"lora-model-{timestamp}"
    else:
        package_name = f"full-model-{timestamp}"
    
    package_dir = os.path.join(run_dir, package_name)
    os.makedirs(package_dir, exist_ok=True)
    
    # 复制最佳模型文件
    if os.path.isdir(best_model_path):
        # 如果是目录，复制整个目录
        shutil.copytree(best_model_path, os.path.join(package_dir, "model"), dirs_exist_ok=True)
    else:
        # 如果是文件，复制到model目录
        model_dir = os.path.join(package_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy2(best_model_path, model_dir)
    
    # 创建并复制标签配置文件
    label_config_path = create_label_config(id2label, package_dir)
    print(f"   ✅ 标签配置文件已创建: {os.path.basename(label_config_path)}")
    
    # 复制训练配置文件
    config_source = os.path.join(run_dir, "config.yaml")
    if os.path.exists(config_source):
        shutil.copy2(config_source, package_dir)
    
    # 创建README文件
    readme_content = f"""# 文本分类模型

## 模型信息
- 基础模型: {config.model_name}
- 使用LoRA: {config.use_lora}
- 问题类型: {config.problem_type}
- 标签数量: {config.num_labels}
- 训练时间: {timestamp}

## 文件说明
- `model/`: 训练好的模型文件
- `label_config.yaml`: 标签配置文件
- `config.yaml`: 训练配置文件
- `README.md`: 本说明文件

## 使用方法
```python
from classifier.models.roberta_classifier import Roberta_Model

# 加载模型
model = Roberta_Model(
    roberta_base='{config.model_name}',
    lora_path='{package_name}.zip',  # 如果使用LoRA
    device='cpu'
)

# 推理
result = model.infer("your text here")
```
"""
    
    readme_path = os.path.join(package_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # 打包为zip文件
    zip_path = shutil.make_archive(package_dir, 'zip', package_dir)
    print(f"   ✅ 模型已打包为: {os.path.basename(zip_path)}")
    
    # 计算文件大小
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)  # MB
    print(f"   📊 打包文件大小: {zip_size:.2f} MB")
    
    return zip_path


def start_training(config: NotebookTrainingConfig, verbose: bool = True, package_model_flag: bool = True) -> Tuple[str, str, Optional[str]]:
    """
    启动训练流程（Notebook友好版本）
    
    Args:
        config: NotebookTrainingConfig实例
        verbose: 是否显示详细信息
        package_model_flag: 是否打包模型
        
    Returns:
        (best_model_path, run_dir, zip_path) 最佳模型路径、运行目录和打包文件路径
    """
    # 获取实际配置
    train_config = config.config
    
    # 获取设备
    device = train_config.get_device()
    if verbose:
        print(f"🖥️  使用设备: {device}")
    
    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(train_config.save_dir, f"notebook_run_{timestamp}")
    train_config.save_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)
    
    if verbose:
        print("=" * 50)
        print("🚀 开始训练流程")
        print("=" * 50)
    
    try:
        # 1. 创建数据加载器
        if verbose:
            print("\n📊 1. 创建数据加载器...")
        
        tokenizer_temp = RobertaTokenizer.from_pretrained(train_config.model_name)
        
        train_loader, valid_loader, test_loader, label2id, id2label = create_data_loaders(
            split_ratio=train_config.data_split_ratio,
            batch_size=train_config.batch_size,
            tokenizer=tokenizer_temp,
            device=device,
            max_length=train_config.max_length,
            shuffle_train=train_config.shuffle_train,
            random_seed=train_config.random_seed,
            max_samples=train_config.max_samples
        )
        
        # 更新配置中的标签数量
        train_config.num_labels = len(label2id)
        
        if verbose:
            print(f"✅ 数据加载完成:")
            print(f"   📝 训练样本: {len(train_loader.dataset)}")
            print(f"   🔍 验证样本: {len(valid_loader.dataset)}")
            print(f"   🧪 测试样本: {len(test_loader.dataset)}")
            print(f"   🏷️  标签数量: {train_config.num_labels}")
        
        # 2. 设置模型
        if verbose:
            print("\n🤖 2. 设置模型和优化器...")
        
        model, optimizer, tokenizer = setup_model(train_config, train_config.num_labels, device)
        
        # 3. 创建训练器
        if verbose:
            print("\n🏋️ 3. 创建训练器...")
        
        trainer = EnhancedTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=valid_loader,
            config=train_config,
            id2label=id2label,
            test_loader=test_loader
        )
        
        # 保存配置
        config_save_path = os.path.join(run_dir, 'config.yaml')
        train_config.save_config(config_save_path)
        
        # 4. 开始训练
        if verbose:
            print("\n🚀 4. 开始训练...")
        
        best_model_path = trainer.train()
        
        # 5. 打包模型 (可选)
        zip_path = None
        if package_model_flag:
            zip_path = package_model(best_model_path, run_dir, id2label, train_config)
        
        if verbose:
            print("\n" + "=" * 50)
            print("🎉 训练完成!")
            print("=" * 50)
            print(f"🎯 最佳模型: {best_model_path}")
            print(f"📁 运行目录: {run_dir}")
            if zip_path:
                print(f"📦 打包文件: {zip_path}")
                print(f"⬇️  可下载: {os.path.basename(zip_path)}")
        
        return best_model_path, run_dir, zip_path
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


def quick_train(
    batch_size: int = 16, 
    num_epochs: int = 12, 
    learning_rate: float = 2e-4,
    use_lora: bool = True,
    test_mode: bool = False,
    package_model_flag: bool = True,
    **kwargs
) -> Tuple[str, str, Optional[str]]:
    """
    快速训练函数（一行代码启动训练）
    
    Args:
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        use_lora: 是否使用LoRA
        test_mode: 是否为测试模式（小规模快速验证）
        package_model_flag: 是否打包模型
        
    Returns:
        (best_model_path, run_dir, zip_path)
    """
    print("🚀 快速训练模式")
    
    # 创建配置
    config = NotebookTrainingConfig()
    
    if test_mode:
        config.small_test()
    else:
        config.quick_setup(
            batch_size=batch_size,
            num_epochs=num_epochs, 
            learning_rate=learning_rate,
            use_lora=use_lora
        )
    
    # 启动训练
    return start_training(config, package_model_flag=package_model_flag)


# 为了方便在notebook中导入，提供一些别名
def create_config():
    """创建训练配置的快捷方式"""
    return NotebookTrainingConfig()


def train(config: NotebookTrainingConfig, package_model_flag: bool = True):
    """训练的快捷方式"""
    return start_training(config, package_model_flag=package_model_flag) 