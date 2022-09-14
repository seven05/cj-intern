#!/usr/bin/env python
# coding: utf-8

# # Confusion matrix 만들기

import json
import pandas as pd
from pandas import json_normalize
import numpy as np
import itertools

# ## ground truth.json 불러오기
# 1. 파일을 읽어서 dataframe에 넣기
# 2. gameTime, label만 추출
# 3. gameTime에서 분:초를 제외한 데이터 삭제
def module_data(confidence_thr):
    with open("./input/groun_truth.json", "r", encoding="utf8") as f:
        contents = f.read() # string 타입
        json_data = json.loads(contents)
    df_json_data_groun_truth = json_normalize(json_data['annotations']) # json파일 -> dataframe에 넣기
    df_json_data_groun_truth = df_json_data_groun_truth.loc[:, ['gameTime', 'label']] # 내가 원하는 데이터만 추출
    df_json_data_groun_truth['gameTime'] = df_json_data_groun_truth.gameTime.str.split(' - ').str[1]  # 1 - , 2 - 와 같은 데이터 삭제

    # df_json_data_groun_truth # 출력


    # ## prediction.json 불러오기
    # 1. 파일을 읽어서 dataframe에 넣기
    # 2. gameTime, label만 추출
    # 3. gameTime에서 분:초 가 아닌 데이터 삭제
    # 4. 특정 threshold를 기준으로 데이터 줄이기
    # 5. gameTime에서 Min, Sec을 추출하여 정렬하기
    #
    # #### 여기서 58분 31초는 받은 데이터 기준으로 더한 것입니다.( prediction의 "1-" 데이터만 보라하셨는데, 이렇게 보는게 데이터가 더 커서 좋을 것 같아 이렇게 진행했습니다).

    # In[4]:


    confidence_threshold = confidence_thr


    with open("./input/prediction.json", "r", encoding="utf8") as f:
        contents = f.read() # string 타입
        json_data = json.loads(contents)
    df = json_normalize(json_data['predictions'])  # json파일 -> dataframe에 넣기
    df = df.loc[:, ['gameTime', 'label', 'confidence', 'half']]   # 내가 원하는 데이터만 추출
    df['gameTime'] = df.gameTime.str.split(' - ').str[1]  # 1 - , 2 - 와 같은 데이터 삭제
    df['confidence'] = df['confidence'].astype(float)     # 내가 원하는 퍼센트 이상만 받기 위해 float형으로 변환
    index_drop = df[df['confidence'] <= confidence_threshold].index         # 특정 값 이하 삭제의 index 모으기
    df.drop(index_drop, inplace=True)                       # 해당 index 삭제

    df['MIN'] = df.gameTime.str.split(':').str[0]  # gameTime에서 분 추출
    df['MIN'] = pd.to_numeric(df['MIN'])
    df['SEC'] = df.gameTime.str.split(':').str[1]  # gameTime에서 초 추출
    df['SEC'] = pd.to_numeric(df['SEC'])

    # predict "1 - "의 분, 초가 58분 31초에서 끊겨서 "2 - "에는 해당 숫자만큼 더하여 진행
    for i in df.index :
        if df['half'][i] == "2" :
            df['MIN'][i] += 58
            df['SEC'][i] += 31


    df_json_data_prediction = df.sort_values(by = ['MIN', 'SEC']) # gameTime을 기준으로 정렬
    df_json_data_prediction = df_json_data_prediction.reset_index(drop=True) # 인덱스 재설정


    df_json_data_groun_truth['MIN'] = df_json_data_groun_truth.gameTime.str.split(':').str[0]  # gameTime에서 분 추출
    df_json_data_groun_truth['MIN'] = pd.to_numeric(df_json_data_groun_truth['MIN'])
    df_json_data_groun_truth['SEC'] = df_json_data_groun_truth.gameTime.str.split(':').str[1]  # gameTime에서 초 추출
    df_json_data_groun_truth['SEC'] = pd.to_numeric(df_json_data_groun_truth['SEC'])


    df_json_data_groun_truth['Time'] = df_json_data_groun_truth['MIN']*60 + df_json_data_groun_truth['SEC']
    df_json_data_groun_truth.drop(['gameTime', 'MIN', 'SEC'], axis=1, inplace=True)

    df_json_data_prediction['Time'] = df_json_data_prediction['MIN']*60 + df_json_data_prediction['SEC']
    df_json_data_prediction.drop(['gameTime', 'MIN', 'SEC', 'half', 'confidence'], axis=1, inplace=True)

    df_json_data_prediction


    # ## confusion matrix

    # #### Counting
    #     인풋 :1. df_json_data_groun_truth -> game_time, label이 있는 dataframe
    #           2. df_json_data_prediction  -> game_time, label이 있는 dataframe
    #           나머지 데이터는 여기부터 필요 X
    #
    #     아웃풋 : cmat에 숫자를 추가한 것

    class_dict = {0 : 'Ball out of play',
                  1 : 'Throw-in',
                  2 : 'Foul',
                  3 : 'Indirect free-kick',
                  4 : 'Clearance',
                  5 : 'Shots on target',
                  6 : 'Shots off target',
                  7 : 'Corner',
                  8 : 'Substitution',
                  9 : 'Kick-off',
                  10 : 'Yellow card',
                  11 : 'Offside',
                  12 : 'Direct free-kick',
                  13 : 'Goal',
                  14 : 'Penalty',
                  15 : 'Red card',
                  16 : 'Start of game',
                  17 : 'End of game',
                  18 : 'Start of replay',
                  19 : 'End of replay',
                  20 : 'Yellow card -> red card',
                  21 : 'Other'}

    class_dict = dict(map(reversed,class_dict.items()))

    cmat = np.array([0]*len(class_dict)*len(class_dict)).reshape((len(class_dict), len(class_dict)))

    total_predict_lable_num = np.array([0]*len(class_dict)).reshape(len(class_dict))
    total_GT_lable_num = np.array([0]*len(class_dict)).reshape(len(class_dict))

    for i in df_json_data_prediction.index:
        position = class_dict.get(df_json_data_prediction["label"][i])
        total_predict_lable_num[position] += 1
    #     idx = class_dict.items(df_json_data_prediction["label"][i])
    # value값으로 key값 찾아서 total_predict_lable_num[label key]에 ++ 해서 저 밑에서 dataFrame에 저장

    for i in df_json_data_groun_truth.index:
        position = class_dict.get(df_json_data_groun_truth["label"][i])
        total_GT_lable_num[position] += 1

    # ### confusion matrix에 counting하기
    time_interval_threshold = 5

    for i in df_json_data_groun_truth.index:
        index = 0
        matching = False

        for j in df_json_data_prediction.index:
            temp = df_json_data_groun_truth["Time"][i] - df_json_data_prediction["Time"][j]
            if temp > time_interval_threshold :
                continue

            elif temp < -1 * time_interval_threshold :
                break

            else : # 시간 차가 양 쪽의 threshold 안에 있다면 == margin 안에 있을 때
                if df_json_data_groun_truth["label"][i] == df_json_data_prediction["label"][j] :
                    matching = True
                    index = j
                    break

        if not matching :
            min_time_diff = abs(df_json_data_groun_truth["Time"][i] - df_json_data_prediction["Time"][0])
            for j in df_json_data_prediction.index:
                temp = abs(df_json_data_groun_truth["Time"][i] - df_json_data_prediction["Time"][j])
                if min_time_diff >= temp:
                    min_time_diff = temp
                    index = j
                    matching = True


        x = class_dict.get(df_json_data_groun_truth["label"][i])
        y = class_dict.get(df_json_data_prediction["label"][index])
        if matching :
            cmat[x,y] += 1

    # print(np.sum(cmat))
    #
    # print(cmat)

    # ## Precision, recall 구하기

    # ## label 별 FP, FN , ... 구하기

    sum_total = np.sum(cmat)
    class_dict = dict(map(reversed,class_dict.items()))

    columns = ["label name","TP", "FN", "FP", "TN"]
    lable_score = pd.DataFrame(columns=columns)

    sum_column = np.array(cmat.sum(axis=0))

    for idx in range(0, len(class_dict)):
        TP = 0
        FN = 0
        for idx_x, value_x in enumerate(cmat[idx]):
            if idx_x == idx :
                TP = value_x
            else:
                FN += value_x

        FP = sum_column[idx] - TP
        TN = sum_total - TP - FN - FP
        lable_score.loc[idx] = [class_dict.get(idx), TP, FN, FP, TN]
    # dataFrame 완성


    # ## precision, recall 계산
    lable_score["Precision"] = lable_score["TP"] / total_predict_lable_num
    lable_score["Recall"] = lable_score["TP"] / total_GT_lable_num

    # print(lable_score)
    precision_list = list(lable_score.loc[:, 'Precision'])
    recall_list = list(lable_score.loc[:, 'Recall'])

    list_return = list(itertools.chain(*zip(precision_list, recall_list)))

    return list_return