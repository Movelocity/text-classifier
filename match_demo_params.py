#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数配置迁移演示：从 demo.py 到新训练流程

这个脚本展示了如何将 reference/demo.py 中的训练参数
精确迁移到新的训练流程中（quick_example.py 和 notebook_utils）
"""

def demo_direct_params():
    """直接使用参数配置（匹配 demo.py）"""
    print("🎯 方法1: 直接参数配置")
    print("=" * 40)
    
    try:
        from classifier.notebook_utils import quick_train
        
        print("📋 使用 quick_train 函数，直接传入参数:")
        print("""
        best_model, run_dir = quick_train(
            batch_size=24,      # 匹配 demo.py: DataLoader(..., batch_size=24, ...)
            num_epochs=12,      # 匹配 demo.py: trainer.train(12)
            learning_rate=2e-4, # 匹配 demo.py: lr=2e-4
            use_lora=True,      # 启用LoRA
            test_mode=False     # 生产模式，不限制数据量
        )
        """)
        
        # 实际执行（可选）
        # best_model, run_dir = quick_train(
        #     batch_size=24,
        #     num_epochs=12,
        #     learning_rate=2e-4,
        #     use_lora=True,
        #     test_mode=False
        # )
        
        print("✅ 这种方法最简单，一行代码就能匹配 demo.py 的主要参数")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")


def demo_config_class():
    """使用配置类（完全匹配 demo.py）"""
    print("\n🎯 方法2: 配置类方式")
    print("=" * 40)
    
    try:
        from classifier.notebook_utils import create_config, train
        
        print("📋 使用配置类，精确控制所有参数:")
        
        # 创建配置
        config = create_config()
        
        # 基础训练参数（匹配 demo.py）
        config.quick_setup(
            batch_size=24,      # demo.py 中的 batch_size
            num_epochs=12,      # demo.py 中的 trainer.train(12)
            learning_rate=2e-4, # demo.py 中的 lr=2e-4
            max_length=256,     # 序列最大长度
            use_lora=True       # 启用LoRA
        )
        
        # LoRA参数（完全匹配 demo.py）
        config.set_lora_config(
            r=48,               # demo.py 中的 r=48
            alpha=24,           # demo.py 中的 lora_alpha=24
            dropout=0.3         # demo.py 中的 lora_dropout=0.3
        )
        
        # 数据分割（匹配常规配置）
        config.set_data_split(
            train=0.8,          # 80% 训练
            valid=0.1,          # 10% 验证
            test=0.1            # 10% 测试
        )
        
        # 设备配置
        config.auto_mode()      # 自动选择GPU/CPU
        
        print("""
        # 代码示例:
        config = create_config()
        config.quick_setup(
            batch_size=24,
            num_epochs=12,
            learning_rate=2e-4,
            use_lora=True
        ).set_lora_config(
            r=48,
            alpha=24,
            dropout=0.3
        ).auto_mode()
        
        # 开始训练
        best_model, run_dir = train(config)
        """)
        
        print("✅ 这种方法提供最大的灵活性，可以精确控制每个参数")
        
        # 实际执行（可选）
        # best_model, run_dir = train(config)
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")


def demo_yaml_config():
    """使用YAML配置文件"""
    print("\n🎯 方法3: YAML配置文件")
    print("=" * 40)
    
    print("📋 使用 demo_config.yaml 配置文件:")
    print("""
    # 1. 使用命令行:
    python main.py --config demo_config.yaml
    
    # 2. 在代码中加载:
    from classifier.config import TrainingConfig
    import yaml
    
    with open('demo_config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = TrainingConfig(**config_dict)
    """)
    
    print("✅ 这种方法适合需要保存和重复使用相同配置的场景")


def show_param_comparison():
    """显示参数对照表"""
    print("\n📊 参数对照表")
    print("=" * 60)
    
    print("┌─────────────────────┬─────────────────────┬─────────────────────┐")
    print("│ 参数                │ demo.py             │ 新训练流程          │")
    print("├─────────────────────┼─────────────────────┼─────────────────────┤")
    print("│ LoRA rank           │ r=48                │ lora_r: 48          │")
    print("│ LoRA alpha          │ lora_alpha=24       │ lora_alpha: 24      │")
    print("│ LoRA dropout        │ lora_dropout=0.3    │ lora_dropout: 0.3   │")
    print("│ 批次大小            │ batch_size=24       │ batch_size: 24      │")
    print("│ 学习率              │ lr=2e-4             │ learning_rate: 2e-4 │")
    print("│ 训练轮数            │ trainer.train(12)   │ num_epochs: 12      │")
    print("│ 优化器              │ AdamW(eps=1e-8)     │ AdamW(eps=1e-8)     │")
    print("│ 模型                │ 'roberta-base'      │ 'roberta-base'      │")
    print("│ 问题类型            │ multi_label_classification │ multi_label_classification │")
    print("│ 设备                │ manual device       │ auto detection      │")
    print("│ 数据加载            │ manual DataLoader   │ auto DataLoader     │")
    print("│ 模型保存            │ manual save         │ auto save & package │")
    print("└─────────────────────┴─────────────────────┴─────────────────────┘")


def main():
    """主函数"""
    print("🚀 demo.py 参数配置迁移演示")
    print("=" * 60)
    
    print("本演示展示如何将 reference/demo.py 中的训练参数")
    print("迁移到新的训练流程 (quick_example.py, notebook_utils)")
    
    demo_direct_params()
    demo_config_class()
    demo_yaml_config()
    show_param_comparison()
    
    print("\n💡 总结:")
    print("1. 最简单: 使用 quick_train() 函数直接传参")
    print("2. 最灵活: 使用 NotebookTrainingConfig 配置类")
    print("3. 最规范: 使用 YAML 配置文件")
    print("4. 所有方法都能精确匹配 demo.py 的参数设置")
    
    print("\n🎯 推荐使用方法2（配置类），因为它:")
    print("  - 支持链式调用，代码简洁")
    print("  - 参数类型安全，有IDE提示")
    print("  - 易于调试和修改")
    print("  - 完全兼容 Jupyter Notebook")


if __name__ == "__main__":
    main() 