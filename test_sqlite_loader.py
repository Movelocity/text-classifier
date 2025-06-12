"""
SQLite数据加载器测试文件

验证新的SQLite数据加载流程是否正常工作，包括：
- 数据库连接
- 数据加载和分割
- 数据集创建
- 数据加载器创建
- 数据格式验证
"""

import os
import sys
import torch
import numpy as np
from transformers import RobertaTokenizer

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classifier.sqlite_data_loader import (
    SQLiteDatasetLoader, 
    SQLiteTextDataset, 
    create_data_loaders,
    parse_labels
)
from sqlite_as_dataset.models import SessionLocal, AnnotationData
from sqlite_as_dataset.services import AnnotationService


def test_database_connection():
    """测试数据库连接"""
    print("=" * 50)
    print("测试1: 数据库连接")
    print("=" * 50)
    
    try:
        # 创建数据库会话
        db = SessionLocal()
        
        # 查询数据总数
        total_count = db.query(AnnotationData).count()
        labeled_count = db.query(AnnotationData).filter(
            AnnotationData.labels.is_not(None),
            AnnotationData.labels != ''
        ).count()
        
        print(f"✅ 数据库连接成功")
        print(f"📊 总记录数: {total_count}")
        print(f"🏷️  有标签记录数: {labeled_count}")
        
        # 显示一些示例数据
        sample_data = db.query(AnnotationData).filter(
            AnnotationData.labels.is_not(None),
            AnnotationData.labels != ''
        ).limit(3).all()
        
        print("\n示例数据:")
        for i, item in enumerate(sample_data, 1):
            print(f"  {i}. 文本: {item.text[:50]}...")
            print(f"     标签: {item.labels}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return False


def test_label_parsing():
    """测试标签解析功能"""
    print("\n" + "=" * 50)
    print("测试2: 标签解析功能")
    print("=" * 50)
    
    test_cases = [
        ("label1, label2, label3", ["label1", "label2", "label3"]),
        ("single_label", ["single_label"]),
        ("  spaced  ,  labels  ", ["spaced", "labels"]),
        ("", []),
        (None, []),
        ("label1,label1,label2", ["label1", "label1", "label2"]),  # 重复标签
    ]
    
    all_passed = True
    
    for test_input, expected in test_cases:
        result = parse_labels(test_input)
        if result == expected:
            print(f"✅ '{test_input}' -> {result}")
        else:
            print(f"❌ '{test_input}' -> {result}, 期望: {expected}")
            all_passed = False
    
    return all_passed


def test_sqlite_dataset_loader():
    """测试SQLite数据集加载器"""
    print("\n" + "=" * 50)
    print("测试3: SQLite数据集加载器")
    print("=" * 50)
    
    try:
        # 创建加载器
        loader = SQLiteDatasetLoader()
        
        # 测试获取标签
        print("获取所有标签...")
        label2id = loader.get_all_labels()
        print(f"✅ 发现 {len(label2id)} 个唯一标签")
        
        # 显示一些标签
        sample_labels = list(label2id.items())[:10]
        print("前10个标签:")
        for label, label_id in sample_labels:
            print(f"  {label_id}: {label}")
        
        # 测试数据分割
        print("\n测试数据分割...")
        train_data, valid_data, test_data = loader.load_data_split(
            split_ratio=[0.7, 0.15, 0.15],
            shuffle=True,
            random_seed=42
        )
        
        print(f"✅ 数据分割成功:")
        print(f"   训练集: {len(train_data)} 样本")
        print(f"   验证集: {len(valid_data)} 样本")
        print(f"   测试集: {len(test_data)} 样本")
        
        # 验证数据格式
        if train_data:
            sample = train_data[0]
            required_keys = ['id', 'text', 'labels']
            has_all_keys = all(key in sample for key in required_keys)
            print(f"✅ 数据格式验证: {'通过' if has_all_keys else '失败'}")
            
            if has_all_keys:
                print(f"   示例数据: ID={sample['id']}, 文本长度={len(sample['text'])}, 标签={sample['labels']}")
        
        loader.close()
        return True, label2id, train_data[:5]  # 返回少量数据用于后续测试
        
    except Exception as e:
        print(f"❌ SQLite数据集加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_sqlite_text_dataset(sample_data, label2id):
    """测试SQLite文本数据集"""
    print("\n" + "=" * 50)
    print("测试4: SQLite文本数据集")
    print("=" * 50)
    
    if not sample_data or not label2id:
        print("❌ 缺少测试数据，跳过此测试")
        return False
    
    try:
        # 创建tokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        device = torch.device('cpu')  # 使用CPU进行测试
        
        # 创建数据集
        print("创建数据集...")
        dataset = SQLiteTextDataset(
            data=sample_data,
            tokenizer=tokenizer,
            device=device,
            label2id=label2id,
            max_length=128
        )
        
        print(f"✅ 数据集创建成功，包含 {len(dataset)} 个样本")
        
        # 测试数据获取
        if len(dataset) > 0:
            sample_item = dataset[0]
            required_keys = ['input_ids', 'attention_mask', 'labels']
            has_all_keys = all(key in sample_item for key in required_keys)
            
            print(f"✅ 数据项格式验证: {'通过' if has_all_keys else '失败'}")
            
            if has_all_keys:
                print(f"   input_ids shape: {sample_item['input_ids'].shape}")
                print(f"   attention_mask shape: {sample_item['attention_mask'].shape}")
                print(f"   labels shape: {sample_item['labels'].shape}")
                print(f"   labels sum: {sample_item['labels'].sum().item()}")
        
        return True
        
    except Exception as e:
        print(f"❌ SQLite文本数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loaders():
    """测试完整的数据加载器创建流程"""
    print("\n" + "=" * 50)
    print("测试5: 完整数据加载器")
    print("=" * 50)
    
    try:
        # 创建tokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        device = torch.device('cpu')
        
        print("创建数据加载器...")
        train_loader, valid_loader, test_loader, label2id, id2label = create_data_loaders(
            split_ratio=[0.7, 0.15, 0.15],
            batch_size=4,  # 小批次用于测试
            tokenizer=tokenizer,
            device=device,
            max_length=128,
            shuffle_train=True,
            random_seed=42
        )
        
        print(f"✅ 数据加载器创建成功")
        print(f"   标签数量: {len(label2id)}")
        print(f"   训练批次数: {len(train_loader)}")
        print(f"   验证批次数: {len(valid_loader)}")
        print(f"   测试批次数: {len(test_loader)}")
        
        # 测试一个批次
        print("\n测试训练批次...")
        for batch in train_loader:
            print(f"   input_ids shape: {batch['input_ids'].shape}")
            print(f"   attention_mask shape: {batch['attention_mask'].shape}")
            print(f"   labels shape: {batch['labels'].shape}")
            print(f"   batch size: {batch['input_ids'].shape[0]}")
            break  # 只测试第一个批次
        
        # 验证标签映射
        print("\n标签映射验证:")
        print("前5个标签:")
        for i in range(min(5, len(id2label))):
            print(f"   {i}: {id2label[i]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_consistency():
    """测试数据一致性"""
    print("\n" + "=" * 50)
    print("测试6: 数据一致性检查")
    print("=" * 50)
    
    try:
        # 两次加载相同配置的数据，验证一致性
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        device = torch.device('cpu')
        
        # 第一次加载
        _, _, _, label2id_1, _ = create_data_loaders(
            split_ratio=[0.8, 0.1, 0.1],
            batch_size=2,
            tokenizer=tokenizer,
            device=device,
            max_length=64,
            random_seed=42
        )
        
        # 第二次加载（相同种子）
        _, _, _, label2id_2, _ = create_data_loaders(
            split_ratio=[0.8, 0.1, 0.1],
            batch_size=2,
            tokenizer=tokenizer,
            device=device,
            max_length=64,
            random_seed=42
        )
        
        # 验证标签映射一致性
        labels_consistent = label2id_1 == label2id_2
        print(f"✅ 标签映射一致性: {'通过' if labels_consistent else '失败'}")
        
        if not labels_consistent:
            print("   标签映射不一致，可能存在随机性问题")
        
        return labels_consistent
        
    except Exception as e:
        print(f"❌ 数据一致性测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("🧪 开始SQLite数据加载器测试")
    print("=" * 60)
    
    test_results = []
    
    # 测试1: 数据库连接
    test_results.append(("数据库连接", test_database_connection()))
    
    # 测试2: 标签解析
    test_results.append(("标签解析", test_label_parsing()))
    
    # 测试3: SQLite数据集加载器
    loader_success, label2id, sample_data = test_sqlite_dataset_loader()
    test_results.append(("SQLite数据集加载器", loader_success))
    
    # 测试4: SQLite文本数据集
    if loader_success:
        dataset_success = test_sqlite_text_dataset(sample_data, label2id)
        test_results.append(("SQLite文本数据集", dataset_success))
    else:
        test_results.append(("SQLite文本数据集", False))
    
    # 测试5: 完整数据加载器
    test_results.append(("完整数据加载器", test_data_loaders()))
    
    # 测试6: 数据一致性
    test_results.append(("数据一致性", test_data_consistency()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"通过: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！SQLite数据加载器工作正常。")
        return True
    else:
        print(f"\n⚠️  {total-passed} 个测试失败，请检查相关功能。")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n✅ 可以安全使用新的SQLite数据加载流程进行训练。")
        sys.exit(0)
    else:
        print("\n❌ 存在问题，请修复后再进行训练。")
        sys.exit(1) 