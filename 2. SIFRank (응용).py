# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from konlpy.tag import Okt, Kkma, Komoran, Hannanum
from collections import defaultdict


def preprocessing():
    model = AutoModel.from_pretrained("klue/bert-base").eval()
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    data = ["텍스트 요약은 텍스트의 관련 정보를 나타내는 여러 가지 방법으로 구성된 광범위한 항목입니다. 이 설명서에 설명된 문서 요약 기능을 통해 추출 텍스트 요약을 사용하여 문서의 요약을 생성할 수 있습니다. 원본 콘텐츠 내에서 가장 중요하거나 관련성 있는 정보를 집합적으로 나타내는 문장을 추출합니다. 이 기능은 너무 길어서 읽을 수 없다고 생각할 수 있는 콘텐츠를 줄이도록 설계되었습니다. 예를 들어 문서, 논문 또는 문서를 주요 문장으로 압축할 수 있습니다."]

    # okt = Okt()
    # words = okt.phrases(data[0])
    #
    okt = Komoran()
    words = okt.nouns(data[0])

    # words = sorted(words, key=lambda x:len(x))
    # clean_words = []
    # for i in range(len(words)):
    #     for j in range(i+1, len(words)):
    #         if words[i] in words[j]:
    #             break
    #     else:
    #         clean_words.append(words[i])
    #
    # words = clean_words

    word_idx = defaultdict(list)

    X = tokenizer(data[0], padding=True, truncation=True, max_length=512, return_tensors='pt')

    word_token = tokenizer(words)['input_ids']
    input_sentence = X['input_ids'][0]

    for wt in word_token:
        st = 0
        temp_id = wt[1:-1]
        decode = tokenizer.decode(temp_id)
        length = len(temp_id)
        while st < len(input_sentence):
            if temp_id == list(input_sentence[st:st+length]):
                word_idx[decode].append(list(range(st, st+length)))
                # print(temp_id, list(input_sentence[st:st + length]), tokenizer.decode(input_sentence[st:st + length]))
            st += 1

    with torch.no_grad():
        output = model(**X)['last_hidden_state'][0]

    return output, words, word_idx


def keyword_extractor(word_idx, sentence_representation, output, extract, plus=False):
    candidate_representation = defaultdict(list)
    max = -float('inf')
    result = ''
    rep = 0
    for key in list(word_idx.keys()):
        if key in list(extract.keys()):
            continue

        word_representation = []
        for word in word_idx[key]:
            word_representation.append(list(torch.sum(output[word], dim=0).cpu().detach().numpy()))

        candidate_representation[key] = word_representation

        word_representation = torch.FloatTensor(word_representation)
        # print(key, word_idx[key], .shape)
        if plus is False:
            sim = F.cosine_similarity(sentence_representation, word_representation)
            max_sim = torch.max(sim)

        else:
            sim = F.cosine_similarity(sentence_representation, word_representation)
            max_sim2 = 0

            for extract_key in list(extract.keys()):
                sim2 = F.cosine_similarity(extract[extract_key], word_representation)
                temp = torch.max(sim2)
                if temp > max_sim2:
                    max_sim2 = temp

            max_sim = 0.5 * torch.max(sim) - 0.5 * max_sim2


        # print(key, word_idx[key], F.cosine_similarity(sentence_representation, word_representation))
        if max_sim > max:
            max = max_sim
            result = key
            rep = word_representation[torch.argmax(sim)]

    return result, rep


def main():
    extract = {}
    output, words, word_idx = preprocessing()
    sentence_representation = output[0]
    rank = []
    for i in range(10):
        key, rep = keyword_extractor(word_idx, sentence_representation, output, extract, plus=True)
        rank.append(key)
        extract[key] = rep
    print(extract.keys())
    print(rank)


if __name__ == '__main__':
    main()