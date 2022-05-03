from matplotlib import pyplot as plt
import constant

def plotHistLossEpoch(num_epoch, loss):
    fig = plt.figure(dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(num_epoch+1), loss, label='loss')
    ax1.legend()
    plt.savefig(constant.saved_path+f'hist_loss_epoch_{num_epoch}.png')
    plt.close()

def plotHistLossEpochGAN(num_epoch_tot, n_iteration_per_epoch, d_loss, g_loss, saved_path):
    fig = plt.figure(dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(num_epoch_tot+1), d_loss[::n_iteration_per_epoch], label='d loss')
    ax1.plot(range(num_epoch_tot+1), g_loss[::n_iteration_per_epoch], label='g loss')
    ax1.legend()
    plt.savefig(saved_path+f'hist_loss_epoch_{num_epoch_tot}.png')
    plt.close()


def plotHistPredEpochGAN(num_epoch_tot, n_iteration_per_epoch, d_real_pred, d_fake_pred, saved_path):
    fig = plt.figure(dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(num_epoch_tot+1), d_real_pred[::n_iteration_per_epoch], label='d real pred moy')
    ax1.plot(range(num_epoch_tot+1), d_fake_pred[::n_iteration_per_epoch], label='d fake pred moy')
    ax1.legend()
    plt.savefig(saved_path+f'hist_pred_epoch_{num_epoch_tot}.png')
    plt.close()