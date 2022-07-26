import pandas as pd
import re
from torch.utils.data import DataLoader, Dataset
from krwordrank.hangle import normalize

class CreateDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item]


def special_token_remover(x):
    # return x
    # x = re.sub('[^A-Za-z0-9가-힣.\s]', ' ', x)
    return re.sub('  ', ' ', re.sub('[\s]', ' ', x.lower()))


def load_data(args):
    # if 'csv' in args.data_path:
    #     df = pd.read_csv(args.data_path, usecols=['본문']).dropna()
    #     data = df['본문'].apply(lambda x:special_token_remover(x)).values
    # else:
    #     with open(args.data_path, 'r', encoding='utf-8') as f:
    #         datas = [x.strip() for x in f.readlines()]
    #     data = [special_token_remover(x) for x in datas]

    data = ["텍스트 요약은 텍스트의 관련 정보를 나타내는 여러 가지 방법으로 구성된 광범위한 항목입니다. 이 설명서에 설명된 문서 요약 기능을 통해 추출 텍스트 요약을 사용하여 문서의 요약을 생성할 수 있습니다. 원본 콘텐츠 내에서 가장 중요하거나 관련성 있는 정보를 집합적으로 나타내는 문장을 추출합니다. 이 기능은 너무 길어서 읽을 수 없다고 생각할 수 있는 콘텐츠를 줄이도록 설계되었습니다. 예를 들어 문서, 논문 또는 문서를 주요 문장으로 압축할 수 있습니다."]
    return data

if __name__ == '__main__':
    import easydict
    args = easydict.EasyDict({
        'data_path': 'data/항공안전문화지표 분석모델.csv',
    })

    data_loader = load_data(args)

    for feature in data_loader:
        print(feature)
        break
