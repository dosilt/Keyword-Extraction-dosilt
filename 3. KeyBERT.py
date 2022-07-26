# -*- coding: utf-8 -*-
import numpy as np
import numpy as np
import itertools

from konlpy.tag import Okt, Komoran
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def preprocessing():
    data = ["텍스트 요약은 텍스트의 관련 정보를 나타내는 여러 가지 방법으로 구성된 광범위한 항목입니다. 이 설명서에 설명된 문서 요약 기능을 통해 추출 텍스트 요약을 사용하여 문서의 요약을 생성할 수 있습니다. 원본 콘텐츠 내에서 가장 중요하거나 관련성 있는 정보를 집합적으로 나타내는 문장을 추출합니다. 이 기능은 너무 길어서 읽을 수 없다고 생각할 수 있는 콘텐츠를 줄이도록 설계되었습니다. 예를 들어 문서, 논문 또는 문서를 주요 문장으로 압축할 수 있습니다."]

    # okt = Okt()
    # candidates = list(set(okt.phrases(data[0])))

    okt = Komoran()
    candidates = list(set(okt.nouns(data[0])))

    model = SentenceTransformer('jhgan/ko-sbert-nli')
    doc_embedding = model.encode([data[0]])
    candidate_embeddings = model.encode(candidates)

    top_n = 10
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    print(keywords)


def main():
    preprocessing()


if __name__ == '__main__':
    main()