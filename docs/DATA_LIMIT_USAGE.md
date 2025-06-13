# 如何限制小规模测试的数据量

## 🎯 概述

为了支持快速测试和原型开发，系统现在支持限制训练数据的数量。这对于：
- 快速验证模型配置
- 在资源有限的环境中测试
- 调试训练流程
- 小规模实验

## 🚀 快速使用

### 方法1：使用内置的小规模测试模式

```python
from classifier.notebook_utils import quick_train

# 自动限制为1万条数据
best_model, run_dir = quick_train(test_mode=True)
```

### 方法2：手动配置数据量限制

```python
from classifier.notebook_utils import create_config, train

# 创建配置并设置数据量限制
config = create_config()
config.set_max_samples(10000)  # 限制为1万条
config.auto_mode()

# 开始训练
best_model, run_dir = train(config)
```

### 方法3：链式配置

```python
from classifier.notebook_utils import create_config, train

# 一行配置多个参数
config = create_config().quick_setup(
    batch_size=8,
    num_epochs=3,
    learning_rate=2e-4
).set_max_samples(5000).auto_mode()  # 限制为5千条

best_model, run_dir = train(config)
```

## 📊 预设配置

### small_test() 模式
```python
config = create_config().small_test()
# 自动设置：
# - 数据量限制：10,000条
# - 批次大小：4
# - 训练轮数：2
# - 最大序列长度：128
```

### 自定义数据量
```python
config = create_config()

# 不同规模的限制
config.set_max_samples(1000)   # 1千条（超小规模）
config.set_max_samples(5000)   # 5千条（小规模）
config.set_max_samples(10000)  # 1万条（中等规模）
config.set_max_samples(None)   # 移除限制（使用全部数据）
```

## 💡 使用建议

### 根据用途选择数据量

| 用途 | 建议数据量 | 说明 |
|------|------------|------|
| 快速调试 | 1,000条 | 验证代码逻辑 |
| 模型验证 | 5,000条 | 测试模型效果 |
| 参数调优 | 10,000条 | 调整超参数 |
| 预训练 | 50,000条+ | 正式训练前的测试 |

### 设备和数据量匹配

```python
# CPU环境：推荐小数据量
config = create_config().set_max_samples(1000).cpu_mode()

# GPU环境：可以使用较大数据量
config = create_config().set_max_samples(10000).gpu_mode()

# 自动选择：根据可用资源自动调整
config = create_config().set_max_samples(5000).auto_mode()
```

## 🔧 技术细节

### 数据采样机制
- 系统会首先从数据库中随机采样指定数量的数据
- 然后按照设定的分割比例（默认8:1:1）分为训练/验证/测试集
- 保证数据的随机性和代表性

### 示例：1万条数据的分割
```
总数据：10,000条
├── 训练集：8,000条 (80%)
├── 验证集：1,000条 (10%)
└── 测试集：1,000条 (10%)
```

## 📝 完整示例

### 示例1：超快速测试
```python
from classifier.notebook_utils import create_config, train
from classifier.model_evaluator import load_model

# 配置超小规模测试
config = create_config()
config.set_max_samples(1000)  # 仅1000条数据
config.small_test()           # 小规模测试设置
config.cpu_mode()             # CPU模式

# 训练
print("开始超快速测试...")
best_model, run_dir = train(config)

# 评估
evaluator = load_model(best_model)
labels = evaluator.predict_single("测试文本")
print(f"预测结果: {labels}")
```

### 示例2：中等规模训练
```python
# 配置中等规模训练
config = create_config()
config.quick_setup(
    batch_size=16,
    num_epochs=5,
    learning_rate=2e-4
).set_max_samples(10000).auto_mode()

# 训练
best_model, run_dir = train(config)
```

### 示例3：动态调整数据量
```python
def progressive_training():
    """渐进式训练：从小数据量开始，逐步增加"""
    
    data_sizes = [1000, 5000, 10000]
    
    for size in data_sizes:
        print(f"\n🚀 开始训练 - 数据量: {size:,}条")
        
        config = create_config()
        config.quick_setup(num_epochs=2)
        config.set_max_samples(size)
        config.auto_mode()
        
        best_model, run_dir = train(config)
        print(f"✅ 完成训练 - 模型: {best_model}")

progressive_training()
```

## ⚠️ 注意事项

1. **数据代表性**：限制数据量时要确保采样的数据具有代表性
2. **标签平衡**：小数据量可能导致某些标签样本不足
3. **模型效果**：数据量减少可能影响模型最终效果
4. **资源使用**：小数据量训练速度快，但可能无法充分利用GPU

## 🚀 运行演示

运行以下命令查看完整演示：

```bash
python quick_example.py
```

选择选项3来查看数据量限制的演示。

## 🔄 配置修改

如需修改默认的数据量限制，可以编辑 `classifier/notebook_utils.py` 中的 `small_test()` 方法：

```python
def small_test(self):
    """小规模测试配置（快速验证）"""
    # 修改这里的数值来改变默认限制
    self.config.max_samples = 5000  # 改为5千条
    # ... 其他配置
```

## 📊 性能对比

| 数据量 | CPU训练时间 | GPU训练时间 | 内存使用 |
|--------|-------------|-------------|----------|
| 1,000条 | ~2分钟 | ~30秒 | ~1GB |
| 5,000条 | ~8分钟 | ~2分钟 | ~3GB |
| 10,000条 | ~15分钟 | ~4分钟 | ~5GB |

*注：实际时间取决于硬件配置和模型参数* 