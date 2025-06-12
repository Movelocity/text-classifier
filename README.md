# 文本分类训练系统

这是一个基于SQLite数据库的文本多分类训练系统，使用RoBERTa模型和LoRA微调技术。

## 功能特性

- ✅ **SQLite数据源**: 直接从SQLite数据库加载训练数据
- ✅ **模块化设计**: 清晰的代码结构，易于维护和扩展
- ✅ **LoRA微调**: 支持高效的LoRA参数微调
- ✅ **完整训练流程**: 数据加载、模型训练、验证、保存一体化
- ✅ **训练跟踪**: 详细的训练日志、指标记录和可视化
- ✅ **早停机制**: 防止过拟合的智能早停
- ✅ **模型管理**: 自动保存最佳模型和检查点

## 项目结构

```
├── main.py                           # 主训练入口
├── test_sqlite_loader.py             # 数据加载器测试
├── classifier/
│   ├── config.py                     # 训练配置
│   ├── sqlite_data_loader.py         # SQLite数据加载器
│   ├── enhanced_trainer.py           # 增强训练器
│   ├── trainer.py                    # 原始训练器
│   ├── data_utils.py                 # 数据处理工具
│   └── models/                       # 模型定义
├── sqlite_as_dataset/                # SQLite数据集模块
└── annotation.db                     # 数据库文件
```

## 快速开始

### 1. 环境准备

安装依赖：
```bash
pip install torch transformers peft scikit-learn matplotlib tqdm pydantic sqlalchemy pyyaml
```

### 2. 测试数据加载

首先运行测试确保数据加载器正常工作：

```bash
python test_sqlite_loader.py
```

如果所有测试通过，说明系统准备就绪。

### 3. 开始训练

#### 使用默认配置训练：
```bash
python main.py
```

#### 使用命令行参数自定义训练：
```bash
python main.py --batch_size 32 --num_epochs 10 --learning_rate 3e-4

# 正常训练并打包
python main.py --config config.yaml

# 训练但不打包
python main.py --config config.yaml --no_package

# 使用LoRA训练并打包
python main.py --use_lora --lora_r 48 --lora_alpha 24
```

#### 使用配置文件训练：
```bash
# 创建配置文件 config.yaml
python main.py --config config.yaml
```

### 4. 监控训练

训练过程中会生成以下文件：
- `models/run_<timestamp>/logs/training.log` - 训练日志
- `models/run_<timestamp>/logs/metrics.json` - 训练指标
- `models/run_<timestamp>/training_curves.png` - 训练曲线图

## 配置参数

### 数据相关
- `data_split_ratio`: 数据分割比例 [train, valid, test]，默认 [0.8, 0.1, 0.1]
- `batch_size`: 批次大小，默认 16
- `max_length`: 最大序列长度，默认 256
- `random_seed`: 随机种子，默认 42

### 模型相关
- `model_name`: 预训练模型名称，默认 "roberta-base"
- `use_lora`: 是否使用LoRA，默认 True
- `lora_r`: LoRA rank，默认 48
- `lora_alpha`: LoRA alpha，默认 24
- `lora_dropout`: LoRA dropout，默认 0.3

### 训练相关
- `num_epochs`: 训练轮数，默认 12
- `learning_rate`: 学习率，默认 2e-4
- `weight_decay`: 权重衰减，默认 0.01
- `warmup_ratio`: 预热比例，默认 0.06

## 命令行参数

```bash
python main.py --help
```

常用参数：
- `--config`: 配置文件路径
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--use_lora`: 启用LoRA
- `--device`: 设备选择 (auto/cpu/cuda)
- `--save_dir`: 模型保存目录

## 输出文件

训练完成后会生成：

```
models/run_<timestamp>/
├── config.yaml                      # 训练配置
├── logs/
│   ├── training.log                 # 训练日志
│   └── metrics.json                 # 训练指标
├── training_curves.png              # 训练曲线
├── checkpoint_epoch_X_<timestamp>/  # 检查点
└── final_model_<timestamp>/         # 最终模型
    ├── pytorch_model.bin            # 模型权重
    ├── config.json                  # 模型配置
    ├── label_config.json            # 标签映射
    ├── training_config.yaml         # 训练配置
    └── metrics.json                 # 最终指标
```

## 数据格式要求

SQLite数据库需要包含以下表结构：

```sql
-- annotation_data 表
CREATE TABLE annotation_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    labels TEXT,  -- 逗号分隔的标签字符串
    UNIQUE(text)
);
```

示例数据：
```
| id | text                    | labels           |
|----|-------------------------|------------------|
| 1  | "这是一个好产品"        | "正面, 产品"      |
| 2  | "服务很差"              | "负面, 服务"      |
| 3  | "价格合理，质量不错"    | "正面, 价格, 质量" |
```

## 训练流程

1. **数据加载**: 从SQLite数据库读取带标签的文本数据
2. **数据预处理**: 分词、编码、创建多热编码标签
3. **数据分割**: 按比例分割为训练、验证、测试集
4. **模型初始化**: 加载预训练RoBERTa模型，应用LoRA配置
5. **训练循环**: 前向传播、反向传播、参数更新
6. **验证评估**: 每epoch后在验证集上评估AUC指标
7. **模型保存**: 保存最佳模型和检查点
8. **结果输出**: 生成训练报告和可视化图表

## 性能监控

系统会自动记录以下指标：
- **训练损失**: 每个batch的训练损失
- **验证损失**: 每个epoch的验证损失
- **验证AUC**: 多标签分类的AUC分数
- **学习率**: 学习率调度变化
- **训练时间**: 每个epoch和总训练时间

## 故障排除

### 常见问题

1. **内存不足**
   - 减小 `batch_size`
   - 减小 `max_length`
   - 使用 `device="cpu"` 强制使用CPU

2. **数据库连接失败**
   - 检查 `annotation.db` 文件是否存在
   - 运行 `test_sqlite_loader.py` 诊断问题

3. **标签数量过多**
   - 检查数据质量，可能存在标签格式问题
   - 考虑标签清理和合并

4. **训练不收敛**
   - 降低学习率
   - 增加 `warmup_ratio`
   - 检查数据质量和标签分布

### 日志分析

查看训练日志：
```bash
tail -f models/run_<timestamp>/logs/training.log
```

查看训练指标：
```bash
cat models/run_<timestamp>/logs/metrics.json | jq .
```

## 扩展功能

### 自定义配置文件

创建 `config.yaml`:
```yaml
# 数据配置
data_split_ratio: [0.8, 0.1, 0.1]
batch_size: 32
max_length: 512

# 模型配置  
model_name: "roberta-large"
use_lora: true
lora_r: 64
lora_alpha: 32

# 训练配置
num_epochs: 20
learning_rate: 1e-4
weight_decay: 0.01
```

### 自定义模型

修改 `setup_model` 函数支持其他预训练模型：
- BERT: `bert-base-uncased`
- DistilBERT: `distilbert-base-uncased`
- ELECTRA: `google/electra-base-discriminator`

### 添加新的评估指标

在 `enhanced_trainer.py` 中扩展 `AUCMetric` 类，添加precision、recall、F1等指标。

## 性能优化

1. **使用更大的批次大小** (如果内存允许)
2. **启用混合精度训练** (添加AMP支持)
3. **使用数据并行** (多GPU训练)
4. **优化数据加载** (增加num_workers)

## 贡献指南

欢迎提交Issue和Pull Request来改进这个训练系统！

## 许可证

本项目遵循MIT许可证。 