import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import easydict
from dataset import load_data
from model import Model
from tqdm import tqdm
import pandas as pd


def init():
    args = easydict.EasyDict({
        'data_path': 'data/항공안전문화지표 분석모델.csv',  # 분석하고자 하는 csv 파일 경로
        'PLM': "klue/bert-base",  # pre-trained language model 이름, huggingface 사이트에서 찾을 수 있음
        'top_n': 10,     # 문장 당 몇개의 단어를 추출할지 결정
        'maxlen': 512,
        'save_path': 'test1_split.csv',
        'device': 'cuda'
    })
    return args


def similarity_matrix(sentence_representation, token_representation, ids, sentence, tok):
    word_representation = []
    word_unit = []
    count = 1
    for cnt, t in enumerate(ids[0][1:-1]):
        decode = tok.decode(t)
        if decode.startswith('##'):
            word_representation[-1] += token_representation[cnt]
            count += 1
            word_unit[-1] += decode[2:]

        else:
            if count != 1:
                word_representation[-1] /= count
                count = 1
            word_representation.append(token_representation[cnt])
            word_unit.append(decode)

    word_representation = np.array(word_representation)
    word_unit = [''.join(x.split()) for x in word_unit]

    for i in range(20):
        sim_matrix = create_sim_matrix(word_representation, sentence_representation, word_unit)
        word_unit = extractor(sim_matrix, word_unit)


def create_sim_matrix(word_representation, sentence_representation, word_unit):
    h = word_representation.shape[0]
    matrix = [[0 for _ in range(h)] for _ in range(h)]
    sim_matrix = np.zeros(shape=(h, h))

    for i in range(h):
        if word_unit[i] is not None and word_unit[i] != '.':
            matrix[i][i] = word_representation[i].astype(np.float16)
            sim_matrix[i][i] = abs(cosine_similarity(sentence_representation.reshape(1, -1), matrix[i][i].reshape(1, -1)))
            for j in range(i):
                if sim_matrix[j][i-1] != 0:
                    matrix[j][i] = matrix[j][i-1] * (i-j) / (i-j+1) + matrix[i][i] / (i-j+1)
                    sim_matrix[j][i] = abs(cosine_similarity(sentence_representation.reshape(1, -1), matrix[j][i].reshape(1, -1)))

    sim_matrix = np.array(sim_matrix)
    return sim_matrix


def extractor(sim_matrix, word_unit):
    row_index = np.argmax(np.max(sim_matrix, axis=1))
    col_index = np.argmax(sim_matrix[row_index])
    print(' '.join(word_unit[row_index:col_index+1]))
    for r in range(row_index, col_index+1):
        word_unit[r] = None
    return word_unit



def main(args):
    data_loader = load_data(args)
    model = Model(args).eval().to(args.device)
    result = []
    for data in tqdm(data_loader):
        with torch.no_grad():
            vector, ids, sentence = model(data)
        vector = vector.cpu().detach().numpy()
        sentence_representation = vector[0]
        word_representation = vector[1:-1]
        similarity_matrix(sentence_representation, word_representation, ids, sentence, model.tokenizer)
        break


if __name__ == '__main__':
    args = init()
    main(args)
