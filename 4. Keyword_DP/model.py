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


if __name__ == '__main__':
    import easydict
    from dataset import load_data

    args = easydict.EasyDict({
        'data_path': 'data/항공안전문화지표 분석모델.csv',
        'PLM': 'monologg/koelectra-base-v3-discriminator',
        'device': 'cpu',
        'maxlen': 512
    })

    data = load_data(args)
    model = Model(args).to(args.device)
    for d in data:
        vector, ids, sentence = model(d)
        print(vector.shape)
        break
