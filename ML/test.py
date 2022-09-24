import mlxtend
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF Algorithm 적용
# TF-IDF => 특정 문서에서 나오는 단어들을 기반으로 vector화 시켜서 유사도를 측정하는 방법론.
# 문장에서 사용되는 단어들의 빈도수를 기반으로 측정하기 때문에, 측정시 사용되는 문장들에서 사용되는 단어의 빈도수가 높으면 그에 따르는 유사도가 높아짐 => 직관적일 수 있음.
# 위에서 설명한 이유로 측정시 빈도수 기반으로 측정하여 직관적인 결과값을 추출할 수 있으나, 메모리 사용률이 매우 높고, 사용되는 문장의 길이가 길어지면 길어질 수록 사용되는
# 메모리, 연산의 수가 기하 급수적으로 증가함. 그렇기 때문에 단순한 문장과 문장들 사이의 측정에 있어서는 효율적이나, 매우 긴 문장들을 비교할 때는 메모리 문제가 있을 수 있다.
# text 기반의 아이템 셋을 비교하기 때문에, 다음과 같이 샘플 데이터를 둔다.
docs = [
    '나는 파이썬을 공부해',
    '너는 파이썬을 공부하니',
    '파이썬과 자바는 다른 언어야',
    '파이썬과 자바는 같은 프로그래밍 언어야'
]
vect=CountVectorizer() #counterVect 객체생성.
countVect=vect.fit_transform(docs) # 문서를 Vectorizer 함수에 적용. 4*9
# print(countVect)
# print(countVect.toarray())
# print(vect.vocabulary_) #vect 값을 dictionary 형태로 출력.
# print(sorted(vect.vocabulary_)) # sorted => 단어 정렬 함수.

counterVect_df = pd.DataFrame(countVect.toarray(), columns=sorted(vect.vocabulary_))
counterVect_df.index = ['문서1', '문서2', '문서3', '문서4']
# print(counterVect_df)

print(cosine_similarity(counterVect_df,counterVect_df)) # 각 문서들의 코사인 유사도를 매트릭스에 수치화하여 표기.


# tfidv = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None).fit(docs)
# tfidv_df = pd.DataFrame(tfidv.transform(docs).toarray())
# # print(tfidv_df)
# print(cosine_similarity(tfidv_df, tfidv_df)) # 위에서 정의한 tf-idf 알고리즘을 cosine 유사도를 적용시켰을 때의 결과값.