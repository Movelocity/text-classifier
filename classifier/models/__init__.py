import torch
from peft import PeftModel, PeftConfig
from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig


def cls_model_with_adapter(base_model_dir, adapter_dir, **kwargs):
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    peft_config.base_model_name_or_path = base_model_dir

    config = AutoConfig.from_pretrained(
        base_model_dir, 
        problem_type="single_label_classification",
        hidden_dropout_prob=0.3,
        **kwargs,  # Adapter 中包含 bert.classifier 的权重，如果形状不对就会报错
    )
    base_model = BertForSequenceClassification.from_pretrained(base_model_dir, config=config, ignore_mismatched_sizes=True)
    adapted_model = PeftModel.from_pretrained(base_model, adapter_dir)
    return adapted_model  # 请在外面进行 to_device, eval


def default_model():
    base_model_dir = r'C:\projects\text-classifier\ckpt\bert_smt95'
    adapter_dir = r'C:\projects\text-classifier\ckpt\bert_ebay36_lora'
    return cls_model_with_adapter(base_model_dir, adapter_dir, num_labels=36)

def default_tokenizer():
    base_model_dir = r'C:\projects\text-classifier\ckpt\bert_smt95'
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    return tokenizer