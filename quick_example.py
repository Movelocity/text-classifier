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
        
        # 这将进行小规模快速训练，适合测试
        print("开始快速训练（测试模式）...")
        best_model_path, run_dir = quick_train(
            batch_size=4,
            num_epochs=1,
            learning_rate=2e-4,
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
        config.small_test()  # 小规模测试
        # config.production_setup()  # 生产环境
        
        # LoRA配置
        print("\n4. LoRA配置:")
        config.set_lora_config(r=32, alpha=16, dropout=0.3)
        
        # 数据分割
        print("\n5. 数据分割:")
        config.set_data_split(train=0.8, valid=0.1, test=0.1)
        
        print("\n✅ 所有配置演示完成!")
        
    except ImportError:
        print("❌ 配置模块导入失败")


if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 完整示例（包含训练）")
    print("2. 仅配置演示")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            main()
        elif choice == "2":
            demo_config_only()
        else:
            print("运行完整示例...")
            main()
            
    except KeyboardInterrupt:
        print("\n👋 用户取消")
    except Exception as e:
        print(f"\n❌ 运行错误: {e}") 