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

    x = np.linspace(1, len(txt1_path_list[:epochs]), len(txt1_path_list[:epochs]), dtype=int)

    plt.plot(x, txt1_path_list[:epochs], label='Darknet', alpha=0.7)
    plt.plot(x, txt2_path_list[:epochs], label='TLA-Darknet', alpha=0.7)

    plt.xlim([0, 17.5])
    plt.xticks(np.arange(0, 17.5, 1))

    plt.scatter(16, txt1_path_list[15], color='g', marker='o', alpha=1, label='epoch=17, loss=0.45 (Darknet)')
    plt.scatter(7, txt2_path_list[6], color='r', marker='o', alpha=1, label='epoch=8, loss=0.41 (TLA-Darknet)')
    plt.axvline(7, linestyle='--', color='g', linewidth=1, alpha=0.4)
    plt.axvline(16, linestyle='--', color='g', linewidth=1, alpha=0.4)

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.4)

    name = txt1_path.split('/')[1] + '_' + txt2_path.split('/')[1]
    plt.savefig(os.path.join(name))


txt1_path = './darknet_416_normal_1009_gcp/darknet_416_normal_1009_gcp_loss_hist.txt'
txt2_path = './darknet_416_tla_1009_gcp/darknet_416_tla_1009_gcp_loss_hist.txt'
txt1_path_list, txt2_path_list = compare_loss(txt1_path, txt2_path)

plot(txt1_path_list, txt2_path_list)