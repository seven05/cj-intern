import time
import soccer_edit
import json
import pandas as pd
from pandas import json_normalize

# 황교훈님 코드 잘 빌렸습니다 ㅎㅎ 
for_for_loop= ["001", "005","010","020", "030","040","050","060","070","080","090","100","110","120","130","140","150"]


columns = ["file_name", "first_start_diff", "first_end_diff", "second_start_diff", "second_end_diff", "time"]
df_diff = pd.DataFrame(columns=columns)
columns_predict = ["file_name", "first_start", "first_end", "second_start", "second_end", "time"]
df_predict = pd.DataFrame(columns=columns_predict)


def read_json(file_name):
    with open(file_name, "r", encoding="utf8") as f:
        contents = f.read()  # string 타입
        json_data = json.loads(contents)
    df_json_data_groun_truth = json_normalize(json_data['annotations'])  # json파일 -> dataframe에 넣기
    df_json_data_groun_truth['gameTime'] = df_json_data_groun_truth.gameTime.str.split(' - ').str[1]  # 1 - , 2 - 와 같은 데이터 삭제

    # df에 순서대로 추가
    # print(df_json_data_groun_truth)  # 출력
    result = []
    for min_sec in df_json_data_groun_truth['gameTime']:
        result.append(int(min_sec.split(":")[0])*60 + int(min_sec.split(":")[1]))

    a = result[0]
    b = result[1]
    c = result[2]
    d = result[3]
    return a, b, c, d

# json 파일 읽어서 분, 초를 초로 변환하기
# gameTime이랑 뭔지 dataFrame에 저장
# game 이름 | 전반 시작 차이 | 전반 종료 차이 | 후반 시작 차이 | 후반 종료 차이 | 걸린 시간
# 차이 = GT - predict (절대값 아님)
# 만들어서 csv 파일로 저장
input_mp4 = "P470472958_EPI0"
input_json = "json/EP"
count = 0
for i in for_for_loop:
    file_name_mp4 = input_mp4 + str(i) + "_01_t35.mp4"
    file_name_json = input_json + str(i) + ".json"
    print("now is going on ", i)
    a_gt, b_gt, c_gt, d_gt = read_json(file_name_json) # x_gt가 모두 int형으로 바꿔어서 받음.

    a, b, c, d = 0, 0, 0, 0
    start = time.time()
    try:
        a, b, c, d = soccer_edit.main(file_name_mp4)
    except:
        print("ERROR IN soccer_edit")
    end = time.time()

    file_name = "EP" + i
    df_diff.loc[count] = [file_name, a - a_gt, b - b_gt, c - c_gt, d - d_gt, end-start]
    df_predict.loc[count] = [file_name, a, b, c, d, end-start]
    count += 1

df_diff.to_csv("result.csv", mode= "w")
df_predict.to_csv("predict.csv", mode="w")