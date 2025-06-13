#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速示例：使用notebook友好的训练和评估功能

这个脚本展示了如何快速训练一个文本分类模型并进行预测。
"""

def main():
    print("🚀 文本分类快速示例")
    print("=" * 50)
    
    try:
        # 导入必要的模块
        from classifier.notebook_utils import quick_train, create_config, train
        from classifier.model_evaluator import load_model
        
        print("✅ 成功导入模块")
        
        # 方法1: 超快速训练（一行代码）
        print("\n🎯 方法1: 超快速训练")
        print("-" * 30)
        
        # 这将进行小规模快速训练，适合测试（数据量限制在1万条以内）
        print("开始快速训练（测试模式）...")
        best_model_path, run_dir, zip_path = quick_train(
            batch_size=24,      # 匹配 demo.py 的 batch_size
            num_epochs=12,      # 匹配 demo.py 的训练轮数
            learning_rate=2e-4, # 匹配 demo.py 的学习率
            use_lora=True,
            test_mode=True
        )
        
        print(f"✅ 训练完成！模型保存在: {best_model_path}")
        
        # 加载模型进行预测
        print("\n🔍 加载模型并进行预测...")
        evaluator = load_model(best_model_path, device="auto")
        
        # 示例文本
        test_texts = [
            "这个产品质量很好，非常满意",
            "服务态度差，等了很久",
            "价格合理，性价比不错"
        ]
        
        print("\n预测结果:")
        for text in test_texts:
            labels = evaluator.predict_single(text, threshold=0.3)
            print(f"  '{text}' -> {labels}")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已正确安装所有依赖包")
        return
    except Exception as e:
        print(f"❌ 训练错误: {e}")
        print("这可能是因为数据库或配置问题")
        
        # 展示配置功能（即使训练失败）
        print("\n🔧 展示配置功能:")
        try:
            config = create_config()
            print("✅ 配置创建成功")
            
            # 展示链式配置
            config.quick_setup(
                batch_size=8,
                num_epochs=3,
                learning_rate=1e-4
            ).set_lora_config(
                r=16,
                alpha=8
            ).auto_mode()
            
            print("✅ 链式配置演示完成")
            
        except Exception as config_e:
            print(f"❌ 配置错误: {config_e}")
    
    print("\n" + "=" * 50)
    print("🎉 示例完成!")
    print("\n💡 下一步:")
    print("1. 查看 notebook_example.ipynb 获得详细示例")
    print("2. 阅读 README_notebook.md 了解完整功能")
    print("3. 在Jupyter Notebook中使用这些功能")


def demo_config_only():
    """演示配置功能（不进行实际训练）"""
    print("🔧 配置功能演示")
    print("=" * 30)
    
    try:
        from classifier.notebook_utils import create_config
        
        # 基础配置
        print("1. 基础配置:")
        config = create_config()
        config.quick_setup(batch_size=16, num_epochs=8)
        
        # 设备配置
        print("\n2. 设备配置:")
        config.auto_mode()  # 或 .cpu_mode() 或 .gpu_mode()
        
        # 预设配置
        print("\n3. 预设配置:")
        config.small_test()  # 小规模测试（自动限制1万条数据）
        # config.production_setup()  # 生产环境
        
        # LoRA配置
        print("\n4. LoRA配置:")
        config.set_lora_config(r=32, alpha=16, dropout=0.3)
        
        # 数据分割
        print("\n5. 数据分割:")
        config.set_data_split(train=0.8, valid=0.1, test=0.1)
        
        # 数据量限制演示
        print("\n6. 数据量限制:")
        config.set_max_samples(5000)  # 限制为5千条
        config.set_max_samples(10000)  # 限制为1万条
        config.set_max_samples(None)   # 移除限制
        
        print("\n✅ 所有配置演示完成!")
        
    except ImportError:
        print("❌ 配置模块导入失败")


def demo_data_limits():
    """演示不同数据量限制的训练"""
    print("📊 数据量限制训练演示")
    print("=" * 40)
    
    try:
        from classifier.notebook_utils import create_config, train
        
        print("演示不同数据量限制的配置:")
        
        # 1. 超小规模测试（1000条）
        print("\n1. 超小规模测试（1000条）:")
        config1 = create_config()
        config1.small_test().set_max_samples(1000).cpu_mode()
        
        # 2. 小规模测试（5000条）
        print("\n2. 小规模测试（5000条）:")
        config2 = create_config()
        config2.quick_setup(batch_size=8, num_epochs=3).set_max_samples(5000).auto_mode()
        
        # 3. 中等规模测试（10000条）
        print("\n3. 中等规模测试（10000条）:")
        config3 = create_config()
        config3.quick_setup(batch_size=16, num_epochs=5).set_max_samples(10000).auto_mode()
        
        print("\n💡 提示:")
        print("- 使用 config.set_max_samples(数量) 来限制数据量")
        print("- 使用 config.small_test() 自动设置1万条限制")
        print("- 使用 config.set_max_samples(None) 移除限制")
        print("- 训练时会优先随机采样指定数量的数据")
        
        print("\n🚀 如需实际训练，请调用:")
        print("   best_model, run_dir = train(config)")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")


def demo_match_demo_py():
    """演示如何完全匹配 demo.py 中的训练参数配置"""
    print("🎯 匹配 demo.py 的参数配置")
    print("=" * 50)
    
    try:
        from classifier.notebook_utils import create_config, train
        
        print("📋 demo.py 中的参数配置:")
        print("  LoRA:")
        print("    - r: 48")
        print("    - alpha: 24") 
        print("    - dropout: 0.3")
        print("  训练:")
        print("    - batch_size: 24")
        print("    - learning_rate: 2e-4")
        print("    - epochs: 12")
        print("    - optimizer: AdamW(eps=1e-8)")
        
        print("\n🔧 在 quick_example.py 中的配置方法:")
        
        # 方法1: 使用 quick_train 直接配置
        print("\n💡 方法1: 使用 quick_train")
        print("```python")
        print("best_model, run_dir = quick_train(")
        print("    batch_size=24,      # 匹配 demo.py")
        print("    num_epochs=12,      # 匹配 demo.py")
        print("    learning_rate=2e-4, # 匹配 demo.py")
        print("    use_lora=True,")
        print("    test_mode=False     # 生产模式")
        print(")")
        print("```")
        
        # 方法2: 使用配置类详细配置
        print("\n💡 方法2: 使用配置类详细配置")
        print("```python")
        config = create_config()
        
        # 基础训练参数（匹配 demo.py）
        config.quick_setup(
            batch_size=24,      # demo.py 中的 batch_size
            num_epochs=12,      # demo.py 中的 trainer.train(12)
            learning_rate=2e-4, # demo.py 中的 lr=2e-4
            use_lora=True
        )
        
        # LoRA参数（完全匹配 demo.py）
        config.set_lora_config(
            r=48,               # demo.py 中的 r=48
            alpha=24,           # demo.py 中的 lora_alpha=24
            dropout=0.3         # demo.py 中的 lora_dropout=0.3
        )
        
        # 设备配置
        config.auto_mode()      # 自动选择设备
        
        print("config = create_config()")
        print("config.quick_setup(")
        print("    batch_size=24,      # 匹配 demo.py")
        print("    num_epochs=12,      # 匹配 demo.py")
        print("    learning_rate=2e-4, # 匹配 demo.py")
        print("    use_lora=True")
        print(").set_lora_config(")
        print("    r=48,               # 匹配 demo.py")
        print("    alpha=24,           # 匹配 demo.py")
        print("    dropout=0.3         # 匹配 demo.py")
        print(").auto_mode()")
        print("")
        print("# 开始训练")
        print("best_model, run_dir = train(config)")
        print("```")
        
        # 方法3: 完整配置（包含所有参数）
        print("\n💡 方法3: 完整配置（包含optimizer eps等）")
        print("```python")
        print("config = create_config()")
        print("# 注意：优化器的 eps=1e-8 参数在 TrainingConfig 中默认已配置")
        print("# 查看 classifier/config.py 第112行附近的 AdamW 配置")
        print("```")
        
        print("\n✅ 配置完成！所有参数已匹配 demo.py")
        
        print("\n📝 参数对照表:")
        print("┌─────────────────┬─────────────────┬─────────────────┐")
        print("│ 参数            │ demo.py         │ quick_example   │")
        print("├─────────────────┼─────────────────┼─────────────────┤")
        print("│ LoRA r          │ 48              │ 48              │")
        print("│ LoRA alpha      │ 24              │ 24              │")
        print("│ LoRA dropout    │ 0.3             │ 0.3             │")
        print("│ batch_size      │ 24              │ 24              │")
        print("│ learning_rate   │ 2e-4            │ 2e-4            │")
        print("│ epochs          │ 12              │ 12              │")
        print("│ optimizer       │ AdamW(eps=1e-8) │ AdamW(eps=1e-8) │")
        print("└─────────────────┴─────────────────┴─────────────────┘")
        
        print("\n🚀 下一步:")
        print("1. 运行上述配置代码")
        print("2. 训练将产生与 demo.py 相同的效果")
        print("3. 模型会自动打包为 .zip 文件")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")


if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 完整示例（包含训练）")
    print("2. 仅配置演示")
    print("3. 数据量限制演示")
    print("4. 匹配 demo.py 参数配置")
    
    try:
        choice = input("请输入选择 (1/2/3/4): ").strip()
        
        if choice == "1":
            main()
        elif choice == "2":
            demo_config_only()
        elif choice == "3":
            demo_data_limits()
        elif choice == "4":
            demo_match_demo_py()
        else:
            print("运行完整示例...")
            main()
            
    except KeyboardInterrupt:
        print("\n👋 用户取消")
    except Exception as e:
        print(f"\n❌ 运行错误: {e}") 