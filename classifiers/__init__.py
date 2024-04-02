import torch
from transformers import BertPreTrainedModel, BertTokenizer, BertModel

from models import BertClassifier, load_model_eval
from trainer import Trainer
# def getlable(n):
#     return id2cnlabel[int(n)]


