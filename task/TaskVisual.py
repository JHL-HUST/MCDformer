import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from task.util import os_makedirs
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def save_training_process(epochs_stats, plot_dir):
    
    os_makedirs(plot_dir)
    fig1 = plt.figure(1)
    plt.plot(epochs_stats.epoch, epochs_stats.lr_list)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("learning rate")
    plt.grid()
    fig1.savefig(os.path.join(plot_dir, 'lr.svg'), format='svg', dpi=150)
    fig1.savefig(os.path.join(plot_dir, 'lr.png'), format='png', dpi=150)
    plt.close()

    fig2 = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_stats.epoch, epochs_stats.train_loss,
             "ro-", label="Train loss")
    plt.plot(epochs_stats.epoch, epochs_stats.val_loss,
             "bs-", label="Val loss")
    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    
    plt.plot(epochs_stats.epoch, epochs_stats.train_acc,
             "ro-", label="Train acc")
    plt.plot(epochs_stats.epoch, epochs_stats.val_acc,
             "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.grid()
    fig2.savefig(os.path.join(plot_dir,'loss_acc.svg'), format='svg', dpi=150)
    fig2.savefig(os.path.join(plot_dir,'loss_acc.png'), format='png', dpi=150)
    plt.show()
    plt.close()

def save_confmat(Confmat_Set, num_snrs, classes, plot_dir ):
    os_makedirs(plot_dir)
    for i, snr in enumerate(num_snrs):
        fig = plt.figure()
        df_cm = pd.DataFrame(Confmat_Set[i],
                             index=classes,
                             columns=classes)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
        heatmap.yaxis.set_ticklabels(
            heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        heatmap.xaxis.set_ticklabels(
            heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        conf_mat_dir = os.path.join(plot_dir, 'conf_mat')
        os.makedirs(conf_mat_dir, exist_ok=True)
        fig.savefig(conf_mat_dir + '/' + f'ConfMat_{snr}dB.svg', format='svg', dpi=150)
        plt.close()
        
def save_snr_acc(Accuracy_list, Confmat_Set, num_snrs, data_name, class_names, plot_dir):
    plt.plot(num_snrs, Accuracy_list)
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Overall Accuracy")
    plt.title(f"Overall Accuracy on {data_name} dataset")
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid()
    acc_dir = os.path.join(plot_dir, 'acc')
    os.makedirs(acc_dir, exist_ok=True)
    plt.savefig(acc_dir + '/' + 'acc.svg', format='svg', dpi=600)
    plt.savefig(acc_dir + '/' + 'acc.png', format='png', dpi=150)
    plt.close()

    Accuracy_Mods = np.zeros((len(num_snrs), Confmat_Set.shape[-1]))

    for i, snr in enumerate(num_snrs):
        Accuracy_Mods[i, :] = np.diagonal(Confmat_Set[i]) / Confmat_Set[i].sum(1)

    for j in range(0, Confmat_Set.shape[-1]):
        plt.plot(num_snrs, Accuracy_Mods[:, j])

    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Overall Accuracy")
    plt.title(f"Overall Accuracy on {data_name} dataset")
    plt.grid()
    plt.legend(class_names)
    plt.savefig(acc_dir + '/' + 'acc_mods.svg', format='svg', dpi=150)
    plt.savefig(acc_dir + '/' + 'acc_mods.png', format='png', dpi=150)
    plt.close()
    
    return Accuracy_Mods