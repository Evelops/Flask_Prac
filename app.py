import numpy as np
import pymysql
import json
import pandas as pd
from flask import Flask, jsonify, render_template, request, redirect, url_for, send_from_directory
from flask_restful import Resource, Api, reqparse, abort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)


def db_connector():
    db = pymysql.connect(host='XXXX',
                         user='XXXX',
                         password='XXXX',
                         db='XXXX',
                         charset='XXXX')
    cursor = db.cursor()
    sql = '''SELECT * FROM korean_food;'''
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


# flask routing
@app.route('/ML', methods=['GET'])
def ML():
    value = request.values['test']  # 유저가 선택한 비선호 음식데이터 셋.
    likeFood = request.values['test2']  # 안드로이드에서 전송한 선호하는 데이터 셋. "," 분기로 나누어서 결과값 2개를 이어서 리턴한다.

    # 사용하고자 하는 csv 데이터 셋을 pandas의 read_csv 메서드를 통해서 불러온다.
    wte_food_data = pd.read_csv('/Users/eomseung-yeol/PycharmProjects/Flask_Prac/wte_food_data.csv')
    # pd.set_option('display.max_columns', 0)
    df = pd.DataFrame(wte_food_data)
    dfInit = df.query('feature.str.contains("")')  # 초기에 비선호 데이터를 추출하기 위한 DataFrame 초기화.
    print("--------------")
    print(dfInit)
    print("--------------")
    print(df)

    # 넘겨받은 값을 기반으로 데이터를 추출한다.
    for i in value.split(","):
        newDfInit = dfInit.query('feature.str.contains(@i)==False')
        print("-------")
        print(i)

    print("csv 필터링 데이터.")
    print(newDfInit)  # csv 파일에서 필터링 한 데이터.

    tfidf = TfidfVectorizer().fit(newDfInit)  # tfidf 알고리즘을 동작시키기 위한 vectorizer 객체.
    tfidf_matrix = tfidf.fit_transform(
        newDfInit.feature)  # 비교해야되는 대상이 각 음식이 가지고 있는 feature 이기 때문에 데이터에서 feature 칼럼만 추출.
    # print("tfidf_matrix")
    # print(tfidf_matrix.shape) #feature에 대해서 tf-idf를 진행한 결과값 => (80,92) 80->feature 갯수 ,92-> 서로 다른 유니크한 속성값의 갯수

    # tf-idf 로직을 수행한 데이터셋에 대해서 cosine 유사도를 구한 결과값을 리턴.
    res_cosine = linear_kernel(tfidf_matrix, tfidf_matrix)
    # print(res_cosine)

    # pandas 시리즈 배열을 통해서, 인덱스가 음식명이고, 인덱스번호가 값인 판다스 시리즈 배열 선언.
    idx = pd.Series(newDfInit.index, index=newDfInit.name).drop_duplicates()

    def food_chk(name, res_cosine=res_cosine):
        # 유저가 선택한 음식 카테고리를 기반으로 인덱스 번호를 추출한다. -> flask 에서 get 요청으로 받은 값을 기반으로, 진행한다.
        getId = idx[name]

        # 모든 음식 데이터에 대해서 유사도를 구한다.
        sims_cosine = list(enumerate(res_cosine[getId]))

        # 코싸인 유사도에 따라 음식데이터를 정렬한다.
        sims_cosine = sorted(sims_cosine, key=lambda x: x[1], reverse=True)

        # 가장 유사한 데이터 2개를 받아온다.
        sims_cosine = sims_cosine[1:3]

        # 가장 유사한 데이터 5개의 인덱스를 받아온다.
        food_idx = [i[0] for i in sims_cosine]

        result_df = newDfInit.iloc[food_idx].copy()
        result_df['score'] = np.round([i[1] for i in sims_cosine], 2) * 100
        # result_df['score'] = result_df['score'].astype(str) + '%'

        # 결과값에서 특징 속성 제거.
        del result_df['feature']
        return result_df

    getFoodList = []  # AOS에서 받아온 유저가 선호하는 음식 리스트를 저장하기 위한 변수.

    # AOS에서 받아온 음식 데이터를 컴마를 기준으로 분기하여 배열에 넣는다.  
    for i in likeFood.split(","):
        getFoodList.append(i)

    print(getFoodList)

    # AOS에서 받아오는 최대 길이가 5 뿐이 안되기 때문에, 임시로 받아오는 데이터의 길이에 따라서, tf-idf 결과 값을 리턴한다.

    if len(getFoodList) == 1:
        # print("len -> "+len(getFoodList))
        res = pd.concat([food_chk(getFoodList[0])], ignore_index=True)
    elif len(getFoodList) == 2:
        # print("len -> "+len(getFoodList))
        res = pd.concat([food_chk(getFoodList[0]), food_chk(getFoodList[1])], ignore_index=True)
    elif len(getFoodList) == 3:
        # print("len -> "+len(getFoodList))
        res = pd.concat([food_chk(getFoodList[0]), food_chk(getFoodList[1]), food_chk(getFoodList[2])],
                        ignore_index=True)
    elif len(getFoodList) == 4:
        # print("len -> "+len(getFoodList))
        res = pd.concat(
            [food_chk(getFoodList[0]), food_chk(getFoodList[1]), food_chk(getFoodList[2]), food_chk(getFoodList[3])],
            ignore_index=True)
    else:
        # print("len -> " + len(getFoodList))
        res = pd.concat(
            [food_chk(getFoodList[0]), food_chk(getFoodList[1]), food_chk(getFoodList[2]), food_chk(getFoodList[3]),
             food_chk(getFoodList[4])], ignore_index=True)

    print(res)

    res = res.drop_duplicates(['name'])  # 이름 컬럼을 기준으로 중복 값 제거.
    # res.sort_values('score',ascending=False)
    resVal = json.dumps(res.to_dict('records'), ensure_ascii=False, indent=4)

    print(json.dumps(res.sort_values('score', ascending=False).to_dict('records'), ensure_ascii=False, indent=4))

    setVal = json.dumps(res.sort_values('score', ascending=False).to_dict('records'), ensure_ascii=False, indent=4)

    return setVal


if __name__ == "__main__":
    app.run(host='0.0.0.0')
