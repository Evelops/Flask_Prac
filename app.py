import pymysql
import json
import pandas as pd
from flask import Flask, jsonify, render_template,request,redirect, url_for, send_from_directory
from flask_restful import Resource, Api,reqparse, abort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
# def db_connector():
#     con = pymysql.connect(host='wte-rds-server.c16ihe3nmzly.ap-northeast-2.rds.amazonaws.com',
#                           user='wte_rds',
#                           password='djaEhd0426',
#                           db='ETY_DB',
#                           charset='utf8'
#                           )
#     # connection으로 부터 Cursor 설정.
#     # db에서 추출한 값을 pandas DataFrame화 시켜야 하기 때문에, 딕셔너리 형태로 리턴해주는 cursor을 사용한다.
#     cursor = con.cursor(pymysql.cursors.DictCursor)
#     sql = '''SELECT * FROM wte_food_data;'''
#     cursor.execute(sql)
#     rows = cursor.fetchall()
#     a = pd.DataFrame(rows)
#
#     # print(a)
#     #
#     # tfidf = TfidfVectorizer().fit(a)  # tfidf 알고리즘을 동작시키기 위한 vectorizer 객체.
#     # tfidf_matrix = tfidf.fit_transform(a.feature)  # 비교해야되는 대상이 각 음식이 가지고 있는 feature 이기 때문에 데이터에서 feature 칼럼만 추출.
#     # # print("tfidf_matrix")
#     # # print(tfidf_matrix.shape) #feature에 대해서 tf-idf를 진행한 결과값 => (80,92) 80->feature 갯수 ,92-> 서로 다른 유니크한 속성값의 갯수
#     #
#     # # tf-idf 로직을 수행한 데이터셋에 대해서 cosine 유사도를 구한 결과값을 리턴.
#     # res_cosine = linear_kernel(tfidf_matrix, tfidf_matrix)
#     # # print(res_cosine)
#     # idx = pd.Series(a.index, index=a.name).drop_duplicates()
#     # print(idx)
#     #
#     # def food_chk(name, res_cosine=res_cosine):
#     #     # 유저가 선택한 음식 카테고리를 기반으로 인덱스 번호를 추출한다. -> flask 에서 get 요청으로 받은 값을 기반으로, 진행한다.
#     #     getId = idx[name]
#     #
#     #     # 모든 음식 데이터에 대해서 유사도를 구한다.
#     #     sims_cosine = list(enumerate(res_cosine[getId]))
#     #
#     #     # 코싸인 유사도에 따라 음식데이터를 정렬한다.
#     #     sims_cosine = sorted(sims_cosine, key=lambda x: x[1], reverse=True)
#     #
#     #     # 가장 유사한 데이터 5개를 받아온다.
#     #     sims_cosine = sims_cosine[1:6]
#     #
#     #     # 가장 유사한 데이터 5개의 인덱스를 받아온다.
#     #     food_idx = [i[0] for i in sims_cosine]
#     #
#     #     result_df = a.iloc[food_idx].copy()
#     #     result_df['score'] = [i[1] for i in sims_cosine]
#     #
#     #     # 결과값에서 특징 속성 제거.
#     #     del result_df['feature']
#     #     return result_df
#
#
#     return str(a)

def db_connector():
    db=pymysql.connect(host='XXX.XXX.XXX.XX',
                      user='XXX',
                      password='XXXX',
                      db='XXXXX',
                      charset='utf8')
    cursor = db.cursor()
    sql='''SELECT * FROM korean_food;'''
    cursor.execute(sql)
    result = cursor.fetchall()
    pdConvert = pd.DataFrame(result)
    db.close()
    return str(pdConvert)


# Flask 객체를 app에 저장.
@app.route('/')
def hello():
    a = db_connector()
    return a

#flask routing
@app.route('/ML', methods=['GET'])
def ML():
    value = request.values['test'] # 유저가 선택한 비선호 음식데이터 셋.
    value2=request.values['test2'] # 안드로이드에서 전송한 선호하는 데이터 셋. "," 분기로 나누어서 결과값 2개를 이어서 리턴한다.

    # 사용하고자 하는 csv 데이터 셋을 pndas의 read_csv 메서드를 통해서 불러온다.
    wte_food_data = pd.read_csv('/Users/eomseung-yeol/PycharmProjects/Flask_Prac/wte_food_data.csv')
    df = pd.DataFrame(wte_food_data)
    dfInit = df.query('feature.str.contains("")')  # 초기에 비선호 데이터를 추출하기 위한 DataFrame 초기화.
    print("--------------")
    print(dfInit)
    print("--------------")
    print(df)

    # 넘겨받은 값을 기반으로 데이터를 추출한다.
    for i in value.split(","):
        dfInit = dfInit.query('feature.str.contains(@i)==False')
        print("-------")
        print(i)


    print("csv 필터링 데이터.")
    print(dfInit)  # csv 파일에서 필터링 한 데이터.

    tfidf = TfidfVectorizer().fit(dfInit)  # tfidf 알고리즘을 동작시키기 위한 vectorizer 객체.
    tfidf_matrix = tfidf.fit_transform(
        dfInit.feature)  # 비교해야되는 대상이 각 음식이 가지고 있는 feature 이기 때문에 데이터에서 feature 칼럼만 추출.
    # print("tfidf_matrix")
    # print(tfidf_matrix.shape) #feature에 대해서 tf-idf를 진행한 결과값 => (80,92) 80->feature 갯수 ,92-> 서로 다른 유니크한 속성값의 갯수

    # tf-idf 로직을 수행한 데이터셋에 대해서 cosine 유사도를 구한 결과값을 리턴.
    res_cosine = linear_kernel(tfidf_matrix, tfidf_matrix)
    # print(res_cosine)

    # pandas 시리즈 배열을 통해서, 인덱스가 음식명이고, 인덱스번호가 값인 판다스 시리즈 배열 선언.
    idx = pd.Series(dfInit.index, index=dfInit.name).drop_duplicates()

    def food_chk(name, res_cosine=res_cosine):
        # 유저가 선택한 음식 카테고리를 기반으로 인덱스 번호를 추출한다. -> flask 에서 get 요청으로 받은 값을 기반으로, 진행한다.
        getId = idx[name]

        # 모든 음식 데이터에 대해서 유사도를 구한다.
        sims_cosine = list(enumerate(res_cosine[getId]))

        # 코싸인 유사도에 따라 음식데이터를 정렬한다.
        sims_cosine = sorted(sims_cosine, key=lambda x: x[1], reverse=True)

        # 가장 유사한 데이터 5개를 받아온다.
        sims_cosine = sims_cosine[1:6]

        # 가장 유사한 데이터 5개의 인덱스를 받아온다.
        food_idx = [i[0] for i in sims_cosine]

        result_df = dfInit.iloc[food_idx].copy()
        result_df['score'] = [i[1] for i in sims_cosine]

        # 결과값에서 특징 속성 제거.
        del result_df['feature']
        return result_df

    res = pd.concat([food_chk("김치찌개")], ignore_index=True)
    print(res)

    res3 = json.dumps(res.to_dict('records'), ensure_ascii=False,indent=4)
    print(json.dumps(res.to_dict('records'), ensure_ascii=False, indent=4))


    return res3


# 최종적으로 결과값을 리턴하는 값.
def resQuery(value):
    # 사용하고자 하는 csv 데이터 셋을 pndas의 read_csv 메서드를 통해서 불러온다.
    wte_food_data = pd.read_csv('/Users/eomseung-yeol/PycharmProjects/Flask_Prac/wte_food_data.csv')
    df = pd.DataFrame(wte_food_data)
    dfInit = df.query('feature.str.contains("")') # 초기에 비선호 데이터를 추출하기 위한 DataFrame 초기화.
    print("--------------")
    print(dfInit)
    print("--------------")
    print(df)

    # 넘겨받은 값을 기반으로 데이터를 추출한다.
    for i in value.split(","):
        dfInit = dfInit.query('feature.str.contains(@i)==False')


    print(dfInit) # csv 파일에서 필터링 한 데이터.

    tfidf = TfidfVectorizer().fit(dfInit)  # tfidf 알고리즘을 동작시키기 위한 vectorizer 객체.
    tfidf_matrix = tfidf.fit_transform(
        dfInit.feature)  # 비교해야되는 대상이 각 음식이 가지고 있는 feature 이기 때문에 데이터에서 feature 칼럼만 추출.
    # print("tfidf_matrix")
    # print(tfidf_matrix.shape) #feature에 대해서 tf-idf를 진행한 결과값 => (80,92) 80->feature 갯수 ,92-> 서로 다른 유니크한 속성값의 갯수

    # tf-idf 로직을 수행한 데이터셋에 대해서 cosine 유사도를 구한 결과값을 리턴.
    res_cosine = linear_kernel(tfidf_matrix, tfidf_matrix)
    # print(res_cosine)

    # pandas 시리즈 배열을 통해서, 인덱스가 음식명이고, 인덱스번호가 값인 판다스 시리즈 배열 선언.
    idx = pd.Series(dfInit.index, index=dfInit.name).drop_duplicates()



    def food_chk(name, res_cosine=res_cosine):
        # 유저가 선택한 음식 카테고리를 기반으로 인덱스 번호를 추출한다. -> flask 에서 get 요청으로 받은 값을 기반으로, 진행한다.
        getId = idx[name]

        # 모든 음식 데이터에 대해서 유사도를 구한다.
        sims_cosine = list(enumerate(res_cosine[getId]))

        # 코싸인 유사도에 따라 음식데이터를 정렬한다.
        sims_cosine = sorted(sims_cosine, key=lambda x: x[1], reverse=True)

        # 가장 유사한 데이터 5개를 받아온다.
        sims_cosine = sims_cosine[1:6]

        # 가장 유사한 데이터 5개의 인덱스를 받아온다.
        food_idx = [i[0] for i in sims_cosine]

        result_df = dfInit.iloc[food_idx].copy()
        result_df['score'] = [i[1] for i in sims_cosine]

        # 결과값에서 특징 속성 제거.
        del result_df['feature']
        return result_df

    res = pd.concat([food_chk("계란김밥")],ignore_index=True)
    return res.to_json(force_ascii=False, orient='records', indent=4)


if __name__ == "__main__":
    app.run(host='0.0.0.0')

