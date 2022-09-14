from fileinput import close
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_score, recall_score, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt


# 변경해야 할 변수
interval = 5
###### each True면 PR CURVE 한장씩 저장
each = False

labels = ['Ball out of play',
            'Throw-in',
            'Foul',
            'Indirect free-kick',
            'Clearance',
            'Shots on target',
            'Shots off target',
            'Corner',
            'Substitution',
            'Kick-off',
              'Yellow card',
              'Offside',
              'Direct free-kick',
              'Goal',
              'Penalty',
              'Red card',
              'Start of game',
              'End of game',
              'Start of replay',
              'End of replay',
              'Yellow card -> red card',
              'Other']

pr_curve = {
    'Ball out of play':[[0]*len(labels) for _ in range(len(labels))],
    'Throw-in':[[0]*len(labels) for _ in range(len(labels))],
    'Foul':[[0]*len(labels) for _ in range(len(labels))],
    'Indirect free-kick':[[0]*len(labels) for _ in range(len(labels))],
    'Clearance':[[0]*len(labels) for _ in range(len(labels))],
    'Shots on target':[[0]*len(labels) for _ in range(len(labels))],
    'Shots off target':[[0]*len(labels) for _ in range(len(labels))],
    'Corner':[[0]*len(labels) for _ in range(len(labels))],
    'Substitution':[[0]*len(labels) for _ in range(len(labels))],
    'Kick-off':[[0]*len(labels) for _ in range(len(labels))],
    'Yellow card':[[0]*len(labels) for _ in range(len(labels))],
    'Offside':[[0]*len(labels) for _ in range(len(labels))],
    'Direct free-kick':[[0]*len(labels) for _ in range(len(labels))],
    'Goal':[[0]*len(labels) for _ in range(len(labels))],
    'Penalty':[[0]*len(labels) for _ in range(len(labels))],
    'Red card':[[0]*len(labels) for _ in range(len(labels))],
    'Start of game':[[0]*len(labels) for _ in range(len(labels))],
    'End of game':[[0]*len(labels) for _ in range(len(labels))],
    'Start of replay':[[0]*len(labels) for _ in range(len(labels))],
    'End of replay':[[0]*len(labels) for _ in range(len(labels))],
    'Yellow card -> red card':[[0]*len(labels) for _ in range(len(labels))],
    'Other':[[0]*len(labels) for _ in range(len(labels))]
}

def ret_conf_mat(confidence):
    with open('C:\\Users\\user\\Desktop\\coop\\confusion_matrix_0819/groun_truth.json', 'r') as f:
        gt_data = json.load(f)
    gt = gt_data["annotations"]

    with open('C:\\Users\\user\\Desktop\\coop\\confusion_matrix_0819/prediction.json', 'r') as f:
        pred_data = json.load(f)

    #### gt도 time sort 
    for i in range(len(gt_data["annotations"])):
        gameTime = gt[i]['gameTime'].split(" - ")[-1]
        min_, sec_ = map(int, gameTime.split(":"))
        gt[i]["Time"] = sec_ + min_*60

    confidence_thr = confidence
    pred_list = []
    #### confidence thresold 보다 낮은 애들 drop
    for i in range(len(pred_data["predictions"])):
        confidence = pred_data["predictions"][i]["confidence"]
        if float(confidence)>=confidence_thr:
            pred_list.append(pred_data["predictions"][i])

    #### for time sort
    for i in range(len(pred_list)):
        gameTime = pred_list[i]["gameTime"].split(" - ")[-1]
        min_, sec = map(int, gameTime.split(":"))
        if pred_list[i]['gameTime'].split(" - ")[0]=='1':
            pred_list[i]["Time"] = min_*60+sec
        else: # 후반 부 애들 58:31 더해주기
            real_min_ = min_ + 58
            real_sec = sec + 31
            if real_sec>60:
                real_min_+=1
                real_sec -= 60
            gameTime = f"2 - {real_min_}:{real_sec}"
            pred_list[i]['gameTime'] = gameTime
            pred_list[i]["Time"] = min_*60+sec + 58 * 60 + 31
    sort_pred = sorted(pred_list, key=lambda k: k['Time'])

    y_true = []
    y_pred = []
    for i in range(len(gt)):
        gt_time = gt[i]['Time']
        gt_matched = False

        for j in range(len(sort_pred)):
            if gt_time-interval <= sort_pred[j]['Time'] <= gt_time+interval and gt[i]['label']==sort_pred[j]['label']:
                y_true.append(gt[i]['label'])
                y_pred.append(sort_pred[j]['label'])
                gt_matched = True

                break

        if not gt_matched and len(sort_pred):
            minarg = 0
            for j in range(len(sort_pred)):
                diff = np.abs(sort_pred[j]['Time']-gt_time)
                if diff<=np.abs(sort_pred[minarg]['Time']-gt_time):
                    minarg = j

            y_true.append(gt[i]['label'])
            y_pred.append(sort_pred[minarg]['label'])

    conf_mat = confusion_matrix(y_true = y_true, y_pred = y_pred, labels=labels)

    sns.set(rc = {'figure.figsize':(15,15)})
    f = sns.heatmap(conf_mat, annot=True, xticklabels=labels, yticklabels=labels, fmt='d', cmap="YlGnBu")
    f.set(title = "Confusion Matrix")
    f.set_xlabel("prediction")
    f.set_ylabel("ground truth")

    plt.tight_layout()
    plt.savefig(f'./conf_mat/confusion_matrix_{round(conf, 2)}.png', dpi = 300)
    plt.clf()

    cnt = int(conf*20)

    for i in range(len(labels)):
        label = labels[i]
        tp = conf_mat[i][i]

        ### precision
        prec = 0
        for k in range(len(sort_pred)):
            if sort_pred[k]['label'] == label:
                prec+=1

        if prec==0:
            precision = 0
        else:
            precision = float(tp/prec)
            #precision = float(conf_mat[i][i])/fps[i]

        ### recall
        rec = 0
        for k in range(len(gt_data['annotations'])):
            if gt_data['annotations'][k]['label'] == label:
                rec+=1

        if rec==0:
            recall = 0
        else:
            recall = float(tp/rec)
            #recall = float(conf_mat[i][i])/fns[i]

        pr_curve[label][cnt][0] = precision
        pr_curve[label][cnt][1] = recall  

    return conf_mat
    

for conf in np.arange(0.0, 1.05, 0.05):
    conf_matrix = ret_conf_mat(conf)

if not each:
    fig, ax = plt.subplots(4, 6, figsize=(48, 32))
    fig.tight_layout()
    ax = ax.reshape(-1)

c=0
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
########## 하나씩 그릴 거면 each True
for label in labels:
    p = []
    r = []
    for i in range(21):
        p.append(pr_curve[label][i][0])
        r.append(pr_curve[label][i][1])

    confi = [n for n in np.arange(0.0, 1.05, 0.05)]
    r_sorted = np.sort(r)
    r_sorted_arg = np.argsort(r)

    p_sorted_by_r = [p[i] for i in r_sorted_arg]
    conf_sorted_by_r = [confi[i] for i in r_sorted_arg]

    
    if each:
        for i in range(21):
            plt.text(r[i], p[i], "{:.2f}".format(conf_sorted_by_r[i]), rotation=60, fontsize=10)

    save_name = label.lower().replace(' ', '_')

    if label=='Yellow card -> red card':
        save_name = "yellow_card_to_red_card"
    

    if each:
        plt.figure()
        plt.plot(r_sorted, p_sorted_by_r, marker='o', color=colors[0])
        plt.plot(r_sorted, p_sorted_by_r, marker='o', color=colors[3], drawstyle='steps-post', linestyle='dashed')
        plt.title(f"{label} PR Curve")
        #plt.ylim(-0.3, 2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(f"pr_curve/{save_name}_pr_curve.png")
        plt.clf()

    if not each:
        ax[c].plot(r_sorted, p_sorted_by_r, marker='o', alpha = 0.7, color=colors[c%len(colors)]) #colors[c%len(colors)]
        #ax[c].plot(r_sorted, p_sorted_by_r, marker='o', alpha = 0.3, color=colors[3], drawstyle='steps-post', linestyle='dashed')
        ax[c].set_title(f"{label}", fontsize=15)
        ax[c].set_ylim(-0.3, 2)
        ax[c].set_xlim(-0.05, 1.05)
        # ax[c].set_xlabel('Recall')
        # ax[c].set_ylabel('Precision')
        c+=1

if not each:
    fig.savefig("./pr_curve/tot.png")
    plt.cla()