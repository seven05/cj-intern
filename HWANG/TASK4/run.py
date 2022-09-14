import Precision_Recall
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

columns = ['Ball_out_of_play_Precision', 'Ball_out_of_play_Recall',
           'Throw_in_Precision', 'Throw_in_Recall',
           'Foul_Precision', 'Foul_Recall',
           'Indirect_free_kick_Precision', 'Indirect_free_kick_Recall',
           'Clearance_Precision', 'Clearance_Recall',
           'Shots_on_target_Precision', 'Shots_on_target_Recall',
           'Shots_off_target_Precision','Shots_off_target_Recall',
           'Corner_Precision', 'Corner_Recall',
           'Substitution_Precision', 'Substitution_Recall',
           'Kick_off_Precision', 'Kick_off_Recall',
           'Yellow_card_Precision', 'Yellow_card_Recall',
           'Offside_Precision', 'Offside_Recall',
           'Direct_free_kick_Precision', 'Direct_free_kick_Recall',
           'Goal_Precision', 'Goal_Recall',
           'Penalty_Precision', 'Penalty_Recall',
           'Red_card_Precision', 'Red_card_Recall',
           'Start_of_game_Precision', 'Start_of_game_Recall',
           'End_of_game_Precision', 'End_of_game_Recall',
           'Start_of_replay_Precision', 'Start_of_replay_Recall',
           'End_of_replay_Precision', 'End_of_replay_Recall',
           'Yellow_card_to_red_card_Precision', 'Yellow_card_to_red_card_Recall',
           'Other_Precision', 'Other_Recall']
thr_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

df = pd.DataFrame(columns=columns, index=thr_list)

# 각 데이터 들을 df의 thr라는 index에 저장
for thr in thr_list:
    df.loc[thr] = Precision_Recall.module_data(thr)

# Nan to zero (Nan -> 0)
df = df.fillna(0)
# print(df)

# df를 각 label마다 모음
Ball_out_of_play = df[['Ball_out_of_play_Precision', 'Ball_out_of_play_Recall']]
Throw_in = df[['Throw_in_Precision', 'Throw_in_Recall']]
Foul = df[['Foul_Precision', 'Foul_Recall']]
Indirect_free_kick = df[['Indirect_free_kick_Precision', 'Indirect_free_kick_Recall']]
Clearance = df[['Clearance_Precision', 'Clearance_Recall']]
Shots_on_target = df[['Shots_on_target_Precision', 'Shots_on_target_Recall']]
Shots_off_target = df[['Shots_off_target_Precision', 'Shots_off_target_Recall']]
Corner = df[['Corner_Precision', 'Corner_Recall']]
Substitution = df[['Substitution_Precision', 'Substitution_Recall']]
Kick_off = df[['Kick_off_Precision', 'Kick_off_Recall']]
Yellow_card = df[['Yellow_card_Precision', 'Yellow_card_Recall']]
Offside = df[['Offside_Precision', 'Offside_Recall']]
Direct_free_kick = df[['Direct_free_kick_Precision', 'Direct_free_kick_Recall']]
Goal = df[['Goal_Precision', 'Goal_Recall']]
Penalty = df[['Penalty_Precision', 'Penalty_Recall']]
Red_card = df[['Red_card_Precision', 'Red_card_Recall']]
Start_of_game = df[['Start_of_game_Precision', 'Start_of_game_Recall']]
End_of_game = df[['End_of_game_Precision', 'End_of_game_Recall']]
Start_of_replay = df[['Start_of_replay_Precision', 'Start_of_replay_Recall']]
End_of_replay = df[['End_of_replay_Precision', 'End_of_replay_Recall']]
Yellow_card_to_red_card = df[['Yellow_card_to_red_card_Precision', 'Yellow_card_to_red_card_Recall']]
Other = df[['Other_Precision', 'Other_Recall']]



titles = ['Ball_out_of_play', 'Throw_in', 'Foul', 'Indirect_free_kick', 'Clearance', 'Shots_on_target', 'Shots_off_target',
          'Corner', 'Substitution', 'Kick_off', 'Yellow_card', 'Offside', 'Direct_free_kick', 'Goal',
          'Penalty', 'Red_card', 'Start_of_game', 'End_of_game', 'Start_of_replay', 'End_of_replay', 'Yellow_card_to_red_card', 'Other']
fig, ax=plt.subplots(3, 8, figsize=(50, 25))

cnt = 0
axe = ax.ravel()
for title in titles:
    # title로 for문 안에서만 사용할 df를 가져옴.
    df_temp = eval(title)
    column_name = title + '_Recall'
    # recall에 대해 정렬
    df_temp = df_temp.sort_values(by=[column_name])
    # 2가지의 graph 그리기
    df_temp.plot(x=title+'_Recall', y=title+'_Precision', ax=axe[cnt], marker='.', legend=None)
    df_temp.plot(x=title+'_Recall', y=title+'_Precision', ax=axe[cnt], marker='.', legend=None, drawstyle='steps-post', linestyle='dashed', color='red')
    cnt += 1

fig.tight_layout(h_pad=5, w_pad=8)
plt.savefig('result.png')
plt.show()
