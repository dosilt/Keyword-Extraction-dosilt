from torch import nn
import torch
from sentence_transformers import SentenceTransformer
from transformers import ElectraModel, ElectraConfig, ElectraTokenizer, AutoTokenizer, AutoModel
import numpy as np


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # self.model = ElectraModel.from_pretrained(args.PLM).to(args.device)
        # self.tokenizer = ElectraTokenizer.from_pretrained(args.PLM)
        self.model = AutoModel.from_pretrained(args.PLM).to(args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.PLM)

        self.device = args.device
        self.maxlen = args.maxlen

    def forward(self, x):
        input_ = self.tokenizer(x, padding=True, truncation=True, max_length=self.maxlen, return_tensors='pt').to(self.device)
        output = self.model(**input_)['last_hidden_state'][0]
        return output, input_['input_ids'], x
