compare.py 함수를 실행시키면 (단, 가장 최근의 version으로 time_module을 import해야함)


|index|file_name|first_start_diff|first_end_diff|second_start_diff|second_end_diff|time|
|------|--------|----------------|--------------|-----------------|---------------|----|
|0|EP001|-17|18|-21|21|18.1669867038726|
|1|EP002|-19|19|-21|16|13.8259286880493|
|2|EP003|-20|17|-16|16|19.1375622749328|


전반전은 넘지않게, 후반전은 넘어야지만 되므로
first_start_diff, second_start_diff의 경우 음수가 떠야 맞는 것이고
first_end_diff, second_end_diff의 경우 양수가 떠야 맞는 정답.


compare.py의 변수중 for_for_loop를 사용하면 모든 동영상을 체크
small_input을 사용하면 1, 2, 3번만 사용하여 체크.

그 결과물은 result.csv


### DIV, CROP 폴더에 있는 파일이 가장 최근 파일입니다.
결과는 아직 시간이 부족해 나오지 않았습니다. 내일 중으로 업로드하겠습니다.
#### 업로드 완료 9/1 11:05