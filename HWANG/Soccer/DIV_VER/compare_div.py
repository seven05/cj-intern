import time
import time_module_crop as time_module
import json
import pandas as pd
from pandas import json_normalize

# for_for_loop_1 = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010",
#                   "011", "012", "013", "014", "015", "016", "017", "018", "019", "020",
#                   "021", "022", "023", "024", "025", "026", "027", "028", "029", "030"]
# for_for_loop_2 = ["031", "032", "033", "034", "035", "036", "037", "038", "039", "040",
#                   "041", "042", "043", "044", "045", "046", "047", "048", "049", "050",
#                   "051", "052", "053", "054", "055", "056", "057", "058", "059", "060"]
# for_for_loop_3 = ["061", "062", "063", "064", "065", "066", "067", "068", "069", "070",
#                   "071", "072", "073", "074", "075", "076", "077", "078", "079", "080",
#                   "081", "082", "083", "084", "085", "086", "087", "088", "089", "090"]
# for_for_loop_4 = ["091", "092", "093", "094", "095", "096", "097", "098", "099", "100",
#                   "101", "102", "103", "104", "105", "106", "107", "108", "109", "110",
#                   "111", "112", "113", "114", "115", "116", "117", "118", "119", "120"]
# for_for_loop_5 = ["121", "122", "123", "124", "125", "126", "127", "128", "129", "130",
#                   "131", "132", "133", "134", "135", "136", "137", "138", "139", "140",
#                   "141", "142", "143", "144", "145", "146", "147", "148", "149", "150"]
titles = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010",
          "011", "012", "013", "014", "015", "016", "017", "018", "019", "020",
          "021", "022", "023", "024", "025", "026", "027", "028", "029", "030",
          "031", "032", "033", "034", "035", "036", "037", "038", "039", "040",
          "041", "042", "043", "044", "045", "046", "047", "048", "049", "050",
          "051", "052", "053", "054", "055", "056", "057", "058", "059", "060",
          "061", "062", "063", "064", "065", "066", "067", "068", "069", "070",
          "071", "072", "073", "074", "075", "076", "077", "078", "079", "080",
          "081", "082", "083", "084", "085", "086", "087", "088", "089", "090",
          "091", "092", "093", "094", "095", "096", "097", "098", "099", "100",
          "101", "102", "103", "104", "105", "106", "107", "108", "109", "110",
          "111", "112", "113", "114", "115", "116", "117", "118", "119", "120",
          "121", "122", "123", "124", "125", "126", "127", "128", "129", "130",
          "131", "132", "133", "134", "135", "136", "137", "138", "139", "140",
          "141", "142", "143", "144", "145", "146", "147", "148", "149", "150"]

wrong_titles = ["016", "038", "047", "064", "087", "120", "132", "135", "138", "142", "144"]

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
input_mp4 = "tving_video_224/P470472958_EPI0"
input_json = "episode_json/EP"
count = 0
for i in titles:
    file_name_mp4 = input_mp4 + str(i) + "_01_t35.mp4"
    file_name_json = input_json + str(i) + ".json"
    print("now is going on ", i)
    a_gt, b_gt, c_gt, d_gt = read_json(file_name_json) # x_gt가 모두 int형으로 바꿔어서 받음.

    file_name = "EP" + i
    start = time.time()
    try:
        a, b, c, d = time_module.main(file_name_mp4)
        end = time.time()
        df_diff.loc[count] = [file_name, a - a_gt, b - b_gt, c - c_gt, d - d_gt, end - start]
        df_predict.loc[count] = [file_name, a, b, c, d, end - start]
    except:
        print("ERROR IN TIME_MODULE.MAIN")
        df_diff.loc[count] = [file_name, -9999, -9999, -9999, -9999, 0]
        df_predict.loc[count] = [file_name, -9999, -9999, -9999, -9999, 0]


    count += 1

df_diff.to_csv("result.csv", mode= "w")
df_predict.to_csv("predict.csv", mode="w")

