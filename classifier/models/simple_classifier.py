import os
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertTokenizer, BertModel, BertForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType, LoraModel

from transformers import AutoConfig

AutoConfig(problem_type='single_label_classification')


class BertClassifier(BertPreTrainedModel):
    """
    上线后请使用 infer() 方法执行文本分类任务
    """
    def __init__(self, config):
        super(BertClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, labels=None):  
        bert_outputs = self.bert(
            input_ids, 
            # token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask
        )
        pooled_output = bert_outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits:torch.Tensor = self.classifier(pooled_output)
        probs = logits.softmax(-1)
        outputs = probs

        if labels is not None:  # 发现传入label则直接算loss，优化训练用的代码结构 for batch in loader: loss, logits = model(**batch)
            loss = self.loss_fct(probs.view(-1, self.config.num_labels), labels.view(-1))
            outputs = (loss, outputs)

        return outputs # loss, logits
    
    def set_tokenizer(self, tokenizer:BertTokenizer):
        self.tokenizer = tokenizer

    def set_class_label(self, id2label):
        self.id2label = id2label

    def infer(self, text:str, return_type:str=''):
        """
        文本分类，可以直接输入文本，返回类型 return_type 可指定为 'dict', 'str', 'int'(默认)
        为简化接口，目前仅支持 batch_size == 1
        """
        max_seq_length = 256
        tokens = self.tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens[:max_seq_length - 2] + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1, 2 choices
        with torch.no_grad():
            pooled_output = self.bert(input_ids)[1]
            logits:torch.Tensor = self.classifier(pooled_output)
            probs = logits.softmax(-1)
            top_probs, top_indices = probs.topk(k=5, sorted=True)
        if return_type == 'dict':
            result_dict = {self.id2label[id.item()]: round(value.item(), 3) for id, value in zip(top_indices[0], top_probs[0])}
            return result_dict
        elif return_type == 'str':
            return self.id2label[top_indices[0, 0].item()]
        else: # 'number'
            return top_indices[0, 0].item()


def load_model_eval(model_path, device=None, eval=True):
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
    tokenizer = BertTokenizer.from_pretrained(model_path)  
    # 如果 tokenizer 的文件不存在，那么请找个tokenizer的对象，执行 tokenizer.save_pretrained('xxx')
    model = BertClassifier.from_pretrained(model_path, use_safetensors=True).to(device)
    if eval:
        print('model in eval mode')
        model.eval()
    model.set_tokenizer(tokenizer)

    # label_file = os.path.join(model_path, 'class_labels.csv')
    # if os.path.exists(label_file):
    #     df = pd.read_csv(label_file)
    #     id2cnlabel = df.set_index('id')['chinese'].to_dict()
    #     model.set_class_label(id2cnlabel)
    return model