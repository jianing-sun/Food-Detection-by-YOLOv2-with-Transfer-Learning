import os
import numpy as np
import matplotlib.pyplot as plt


def compare_loss(txt1_path, txt2_path):

    txt1_path_list = []
    with open(txt1_path, 'r') as txt1_loss:
        for i, line in enumerate(txt1_loss):
            loss = float(line.rstrip('\n'))
            txt1_path_list.append(loss)

    txt2_path_list = []
    with open(txt2_path, 'r') as txt2_loss:
        for i, line in enumerate(txt2_loss):
            loss = float(line.rstrip('\n'))
            txt2_path_list.append(loss)

    return txt1_path_list, txt2_path_list


def plot(txt1_path_list, txt2_path_list):
    epochs = min(len(txt1_path_list), len(txt2_path_list))
    print('-> txt1 has {} epochs data, txt2 has {} epochs data.'.format(len(txt1_path_list), len(txt2_path_list)))
    print('-> hence we gonna use first {} epochs data'.format(epochs))

    print(txt1_path_list[:epochs])
    print(txt2_path_list[:epochs])

    x = np.linspace(0, len(txt1_path_list[:epochs]) - 1, len(txt1_path_list[:epochs]), dtype=int)

    plt.plot(x, txt1_path_list[:epochs], label='MobileNet', alpha=0.7)
    plt.plot(x, txt2_path_list[:epochs], label='TLA-MobileNet', alpha=0.7)
    # plt.axhline(txt2_path_list[8], linestyle='--', color='g', linewidth=1, alpha=0.5)

    plt.scatter(8, txt2_path_list[8], color='r', marker='o', alpha=1, label='epoch=8, loss=0.30 (TLA-MobileNet)')
    plt.scatter(16, txt1_path_list[16], color='g', marker='o', alpha=1, label='epoch=16, loss=0.45 (MobileNet)')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True, alpha=0.4)

    name = txt1_path.split('/')[1] + '_' + txt2_path.split('/')[1]
    plt.savefig(os.path.join(name))


txt1_path = './tf_log_1003_mn224_normal/loss.txt'
txt2_path = './tf_log_mn224_tla_1004_gcp_1/mn224_tla_1004_gcp_loss_hist.txt'
txt1_path_list, txt2_path_list = compare_loss(txt1_path, txt2_path)

plot(txt1_path_list, txt2_path_list)