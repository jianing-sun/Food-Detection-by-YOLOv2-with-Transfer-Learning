import os

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


import numpy as np


def plot(txt1_path_list, txt2_path_list):
    epochs = min(len(txt1_path_list), len(txt2_path_list))
    print('-> txt1 has {} epochs data, txt2 has {} epochs data.'.format(len(txt1_path_list), len(txt2_path_list)))
    print('-> hence we gonna use first {} epochs data'.format(epochs))

    print(txt1_path_list[:epochs])
    print(txt2_path_list[:epochs])

    x = np.linspace(1, len(txt1_path_list[:epochs]), len(txt1_path_list[:epochs]), dtype=int)

    plt.plot(x, txt1_path_list[:epochs], label='ResNet18', alpha=0.7, color='k', linewidth=0.7)
    plt.plot(x, txt2_path_list[:epochs], label='TLA-MobileNetV2', alpha=0.7, color='r', linewidth=0.7)

    plt.xlim([0, 17.5])
    plt.xticks(np.arange(0, 17.5, 1))

    # plt.scatter(16, txt1_path_list[12], color='k', marker='o', alpha=1,
    #             label='epoch=16, loss=0.31 (ResNet18)', linewidth=0)
    # plt.scatter(7, txt2_path_list[6], color='r', marker='o', alpha=1,
    #             label='epoch=7, loss=0.34 (TLA-ResNet18)', linewidth=0)
    # plt.axvline(7, linestyle='--', color='b', linewidth=1, alpha=0.2)
    # plt.axvline(16, linestyle='--', color='b', linewidth=1, alpha=0.2)

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.4)

    name = txt1_path.split('/')[1] + '_' + txt2_path.split('/')[1]
    plt.savefig(os.path.join(name))


def plot_map2iou():
    mn1_normal = [0.6825, 0.6526, 0.4879, 0.4764, 0.2132, 0.0]
    mn1_tla = [0.7637, 0.7198, 0.6666, 0.5900, 0.3224, 0.0]

    mn2_normal = [0.5951, 0.5383, 0.4893, 0.3856, 0.1555, 0.0]
    mn2_tla = [0.7829, 0.7438, 0.6816, 0.5763, 0.3015, 0.0]

    rn18_tla = [0.6166, 0.5860, 0.5755, 0.5039, 0.1987, 0.0]
    rn18_normal = [0.5319, 0.5301, 0.4927, 0.4175, 0.1542, 0.0]

    iou = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    plt.plot(iou, rn18_tla, label='TLA-ResNet18', color='r', linewidth=0.7)
    plt.plot(iou, rn18_normal, label='ResNet18', color='k', linewidth=0.7)
    plt.xlabel('IoU')
    plt.ylabel('mAP')

    plt.xlim([0.5, 1.0])
    plt.xticks(np.arange(0.5, 1.01, 0.1))
    plt.ylim([0, 0.9])
    plt.yticks(np.arange(0, 1.0, 0.1))

    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.5)

    plt.savefig('../evaluation_results/mAP2IoU_rn18.png')


def plot_comparison():
    mn1_normal = [0.6825, 0.6526, 0.4879, 0.4664, 0.2132, 0.0]
    mn1_tla = [0.7637, 0.7198, 0.6666, 0.5900, 0.3224, 0.0]
    print(np.array(mn1_tla).sum() / 5)
    mn1_freeze = [0.4583, 0.4501, 0.4054, 0.3140, 0.1105, 0.0]
    iou = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    plt.scatter(iou, mn1_normal, color='r', marker='*', label='Baseline, BaseB')
    plt.scatter(iou, mn1_tla, color='g', marker='^', label='Transfer, A11B+')
    plt.scatter(iou, mn1_freeze, color='b', marker='d', label='Frozen, A11B')

    plt.xlabel('IoU')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.savefig('../evaluation_results/mn1_comparison.png')


### draw comparison between tla and normal net
# txt1_path = './mn2_normal_1018_gcp/mn2_normal_1018_gcp_loss_hist.txt'
# txt2_path = './mnv2_224_1007_tla/mnv2_224_tla_1007_gcp_loss_hist.txt'
# txt1_path = './tf_log_1003_mn224_normal/loss.txt'
# txt2_path = './tf_log_mn224_tla_1004_gcp/mn224_tla_1004_gcp_loss_hist.txt'
# txt1_path = './rn18_normal_Oct22/rn18_normal_Oct22_loss_hist_list.txt'
# txt2_path = './rn18_tla_Oct22/rn18_tla_Oct22_loss_hist_list.txt'
# # txt1_path_list, txt2_path_list = compare_loss(txt1_path, txt2_path)
#
# # plot(txt1_path_list, txt2_path_list)
#
#
# ### plot mAP with IoU
# # plot_map2iou()
#
#
plot_comparison()

# mn1_normal = [0.6825, 0.6526, 0.4879, 0.4764, 0.2132, 0.0]
# mn1_tla = [0.7637, 0.7198, 0.6666, 0.5900, 0.3224, 0.0]
#
# mn2_normal = [0.5951, 0.5383, 0.4893, 0.3856, 0.1555, 0.0]
# mn2_tla = [0.7829, 0.7438, 0.6816, 0.5763, 0.3015, 0.0]
#
# rn18_tla = [0.6166, 0.5860, 0.5755, 0.5039, 0.1987, 0.0]
# rn18_normal = [0.5319, 0.5301, 0.4927, 0.4175, 0.1542, 0.0]
#
# print((np.array(mn1_tla) - np.array(mn1_normal)).sum() / 5)
# print((np.array(mn2_tla) - np.array(mn2_normal)).sum() / 5)
# print((np.array(rn18_tla) - np.array(rn18_normal)).sum() / 5)
