# Notebook 友好的文本分类系统

这个文档介绍如何在 Jupyter Notebook 中使用我们的文本分类系统进行快速训练和评估。

## 🚀 快速开始

### 1. 一行代码训练
```python
from classifier.notebook_utils import quick_train
from classifier.model_evaluator import load_model

# 快速训练（测试模式）
best_model, run_dir = quick_train(test_mode=True)

# 加载模型
evaluator = load_model(best_model)

# 预测
labels = evaluator.predict_single("这个产品质量很好")
print(labels)
```

### 2. 链式配置训练
```python
from classifier.notebook_utils import create_config, train

# 创建并配置
config = create_config().quick_setup(
    batch_size=16,
    num_epochs=8,
    learning_rate=2e-4
).set_lora_config(
    r=32,
    alpha=16
).auto_mode()

# 开始训练
best_model, run_dir = train(config)
```

## 📋 功能特性

### NotebookTrainingConfig 类

提供 notebook 友好的配置方式：

```python
config = create_config()

# 基础配置
config.quick_setup(batch_size=16, num_epochs=8, learning_rate=2e-4)

# 模型配置
config.set_model("roberta-base")
config.set_lora_config(r=48, alpha=24, dropout=0.3)

# 数据配置
config.set_data_split(train=0.8, valid=0.1, test=0.1)

# 设备配置
config.auto_mode()  # 自动选择
config.gpu_mode()   # 强制GPU
config.cpu_mode()   # 强制CPU

# 预设配置
config.small_test()      # 小规模测试
config.production_setup()  # 生产环境

# 保存目录
config.set_save_dir("my_models")
```

### ModelEvaluator 类

强大的模型评估和推理功能：

```python
from classifier.model_evaluator import ModelEvaluator, load_model

# 加载模型
evaluator = load_model("path/to/model", device="auto")

# 查看模型信息
evaluator.print_model_info()

# 单条预测
labels = evaluator.predict_single("文本内容", threshold=0.5)

# 详细预测（包含概率）
result = evaluator.predict_single(
    "文本内容", 
    threshold=0.3, 
    return_probabilities=True
)
print(result['predicted_labels'])
print(result['all_scores'])

# 批量预测
texts = ["文本1", "文本2", "文本3"]
batch_results = evaluator.predict_batch(
    texts=texts,
    threshold=0.5,
    batch_size=32,
    show_progress=True
)
```

## 🎯 使用场景

### 场景1：快速实验
```python
# 快速测试想法
best_model, _ = quick_train(
    batch_size=8,
    num_epochs=2,
    test_mode=True  # 小规模快速验证
)

evaluator = load_model(best_model)
labels = evaluator.predict_single("测试文本")
```

### 场景2：生产训练
```python
# 完整训练流程
config = create_config().production_setup().gpu_mode()
best_model, run_dir = train(config)

# 评估模型
evaluator = load_model(best_model)
evaluator.print_model_info()
```

### 场景3：模型比较
```python
# 加载多个模型进行比较
models = ["model1", "model2", "model3"]
test_text = "比较文本"

for model_path in models:
    evaluator = load_model(model_path)
    labels = evaluator.predict_single(test_text, return_probabilities=True)
    print(f"{model_path}: {labels['all_scores']}")
```

### 场景4：批量处理
```python
# 处理大量文本
import pandas as pd

# 读取数据
df = pd.read_csv("texts.csv")
texts = df['text'].tolist()

# 批量预测
evaluator = load_model("best_model")
results = evaluator.predict_batch(
    texts=texts,
    threshold=0.5,
    batch_size=64,
    show_progress=True
)

# 保存结果
df['predicted_labels'] = [','.join(labels) for labels in results]
df.to_csv("predictions.csv", index=False)
```

## ⚙️ 配置选项

### 训练配置
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `batch_size` | 批次大小 | 16 |
| `num_epochs` | 训练轮数 | 12 |
| `learning_rate` | 学习率 | 2e-4 |
| `max_length` | 最大序列长度 | 256 |
| `use_lora` | 是否使用LoRA | True |

### LoRA配置
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `r` | LoRA rank | 48 |
| `alpha` | LoRA alpha | 24 |
| `dropout` | LoRA dropout | 0.3 |

### 预设模式
- **`small_test()`**: 小规模测试（batch_size=4, epochs=2, max_length=128）
- **`production_setup()`**: 生产环境（batch_size=32, epochs=20, max_length=512）
- **`cpu_mode()`**: CPU模式（自动调整批次大小）

## 📊 评估功能

### 预测选项
```python
# 基础预测
labels = evaluator.predict_single("文本", threshold=0.5)

# 详细预测
result = evaluator.predict_single(
    "文本", 
    threshold=0.3,
    return_probabilities=True
)

# 返回结果包含：
# - text: 输入文本
# - predicted_labels: 预测标签列表
# - all_scores: 所有标签的概率分数
# - threshold: 使用的阈值
```

### 批量处理
```python
# 批量预测，支持进度条
results = evaluator.predict_batch(
    texts=["文本1", "文本2"],
    threshold=0.5,
    batch_size=32,
    show_progress=True
)
```

## 🔧 高级用法

### 交互式预测
```python
def interactive_predict(evaluator):
    while True:
        text = input("输入文本 (quit退出): ")
        if text.lower() in ['quit', 'q']:
            break
        
        result = evaluator.predict_single(
            text, 
            threshold=0.3, 
            return_probabilities=True
        )
        
        print(f"预测标签: {result['predicted_labels']}")
        
        # 显示前3个最高分数
        sorted_scores = sorted(
            result['all_scores'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        print(f"前3个分数: {sorted_scores[:3]}")

# 使用
interactive_predict(evaluator)
```

### 模型信息查看
```python
# 详细模型信息
evaluator.print_model_info()

# 包含：
# - 模型路径
# - 设备信息
# - 标签数量
# - 参数统计
# - 训练指标（如果有）
# - 所有标签列表
```

### 结果保存
```python
import pandas as pd

# 预测结果转DataFrame
results_data = []
for text in texts:
    result = evaluator.predict_single(text, return_probabilities=True)
    results_data.append({
        'text': text,
        'predicted_labels': ','.join(result['predicted_labels']),
        'max_score': max(result['all_scores'].values()),
        'num_labels': len(result['predicted_labels'])
    })

df = pd.DataFrame(results_data)
df.to_csv('predictions.csv', index=False)
```

## 🚨 注意事项

1. **设备选择**: 使用 `auto_mode()` 自动选择最佳设备
2. **内存管理**: 大批量预测时适当调整 `batch_size`
3. **模型路径**: 确保模型路径包含 `label_config.json` 文件
4. **依赖检查**: LoRA模型需要安装 `peft` 库

## 📝 示例 Notebook

查看 `notebook_example.ipynb` 获得完整的使用示例，包括：
- 快速训练
- 模型配置
- 批量预测
- 结果可视化

## 🎉 快速命令备忘

```python
# 导入
from classifier.notebook_utils import quick_train, create_config, train
from classifier.model_evaluator import load_model

# 最简单的使用方式
best_model, _ = quick_train(test_mode=True)
evaluator = load_model(best_model)
labels = evaluator.predict_single("你的文本")

# 自定义配置
config = create_config().quick_setup(batch_size=16).auto_mode()
best_model, _ = train(config)

# 批量预测
results = evaluator.predict_batch(["文本1", "文本2"])
```

开始使用吧！🚀 