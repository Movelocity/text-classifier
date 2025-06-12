import torch
import zipfile
from peft import PeftModel
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import numpy as np 
import os
import yaml

def remap_labels(name2conf:dict, hidden_labels:list, target_label:str):
    highest_confidence = name2conf.get(target_label, 0)

    for k, v in name2conf.items():
        if k in hidden_labels:
            if v > highest_confidence:
                highest_confidence = v

    for k in hidden_labels:
        if k in name2conf.keys():
            name2conf.pop(k)

    if highest_confidence > 0:
        name2conf[target_label] = highest_confidence

class Roberta_Model:
    def __init__(self, base_dir, lora_name=None, lora_path=None, device:torch.device="cpu"):
        if lora_path is None and lora_name is None:
            raise ValueError('请提供其中一个参数：lora_name lora_path')
        if lora_path is not None:
            peft_dir = lora_path
        else:
            peft_dir = os.path.join(base_dir, lora_name)
        print(f"load lora model: {peft_dir}")

        # Unzip if peft_dir is a zip file
        if peft_dir.endswith('.zip'):
            peft_dir = self._unzip_model(peft_dir)

        # Open the YAML file
        self.id2label = {}
        label_config_dir = os.path.join(peft_dir, 'label_config.yaml')
        # 标签配置文件结构
        # id2labels:
        #   数字: 字符串标签
        #   ...
        # hide_sub_labels:
        #   字符串标签（显示用）:
        #     - 想要隐藏的子标签（预测时会显示为显示用的标签）
        #     - ...
        if os.path.exists(label_config_dir):
            with open(label_config_dir, 'r') as f:
                # Load the YAML data
                label_config = yaml.safe_load(f)
                self.id2label = label_config['id2label']
                self.hide_sub_labels = label_config.get('hide_sub_labels', {})
        else:
            raise ValueError(f'未发现标签配置文件: {label_config_dir}')

        inference_model = RobertaForSequenceClassification.from_pretrained(
            base_dir, num_labels=len(self.id2label.keys())).to(device).eval()
        inference_model.config.problem_type = "multi_label_classification"
        self.tokenizer = RobertaTokenizer.from_pretrained(base_dir)

        inference_model = PeftModel.from_pretrained(inference_model, peft_dir)  # Load the Lora model
        self.inference_model = inference_model.to(device).eval()
        print(f'load model `{base_dir}` with lora `{lora_name}`')

    def infer(self, text, threshold=0.3):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = self.inference_model(**encoded_input)
            probs = output.logits[0].sigmoid().cpu().numpy().astype(np.float64)  # 单条数据推理，结果去掉 batch 维度

        result_dict = {}
        top_i = np.argmax(probs)
        result_dict[self.id2label[top_i]] = probs[top_i]  # 保底一个意图

        for i, v in enumerate(probs):
            if v > threshold:
                result_dict[self.id2label[i]] = v
        # TODO: 使用 hide_sub_labels 来决定哪些要重映射

        for key, value in self.hide_sub_labels.items():
            remap_labels(result_dict, hidden_labels=value, target_label=key)

        return result_dict
    
    def _unzip_model(self, zip_path):
        """Unzips the specified zip file into a temporary directory."""
        temp_dir = os.path.join(os.getcwd(), '.temp')
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        else:
            # Clear existing contents
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)

        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        print(f'Unzipped model to `{temp_dir}`')
        return temp_dir


def load_model(base_dir, lora_name, device=None):
    """
    model_path: 模型目录，其下应包含
        class_labels.csv  model.safetensors        tokenizer_config.json
        config.json       special_tokens_map.json  vocab.txt
    
    返回的模型默认为推理模式, 已整合 torch.no_grad()

    >>> model = load_model('路径')
    >>> text = "how many shoes are in the store?"
    >>> model.infer(text, return_type='str')
    "库存咨询"
    """
    #  model.safetensor, 
    if device == None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 默认设备
    return Roberta_Model(base_dir, lora_name, device)
    
