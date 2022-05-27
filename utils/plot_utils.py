from matplotlib import pyplot as plt
import constants.constants as constants
from matplotlib.ticker import MaxNLocator

def plotHistLossEpoch(num_epoch, loss, t_loss=None):
    fig = plt.figure(dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(num_epoch+1), loss, label='loss')
    if(t_loss != None):
        ax1.plot(range(num_epoch+1), t_loss, label='test loss')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    plt.savefig(constants.saved_path+f'loss_epoch_{num_epoch}.png')
    plt.close()

def plotHistAllLossEpoch(num_epoch, loss_eye, loss_pose_t, loss_pose_r, loss_au, loss):
    fig = plt.figure(dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(num_epoch+1), loss_eye, label='loss gaze')
    ax1.plot(range(num_epoch+1), loss_pose_t, label='loss pose t')
    ax1.plot(range(num_epoch+1), loss_pose_r, label='loss pose r')
    ax1.plot(range(num_epoch+1), loss_au, label='loss AU')
    ax1.plot(range(num_epoch+1), loss, label='loss')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    plt.savefig(constants.saved_path+f'all_loss_epoch_{num_epoch}.png')
    plt.close()

def plotHistLossEpochGAN(num_epoch, d_loss, t_loss, g_loss=None):
    fig = plt.figure(dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(num_epoch+1), d_loss, label='discriminator loss')
    ax1.plot(range(num_epoch+1), g_loss, label='generator loss')
    if(t_loss != None):
        ax1.plot(range(num_epoch+1), t_loss, label='test loss')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    plt.savefig(constants.saved_path+f'loss_epoch_{num_epoch}.png')
    plt.close()


def plotHistPredEpochGAN(num_epoch, d_real_pred, d_fake_pred):
    fig = plt.figure(dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(num_epoch+1), d_real_pred, label='discriminator real prediction')
    ax1.plot(range(num_epoch+1), d_fake_pred, label='discriminator fake prediction')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Label")
    ax1.legend()
    plt.savefig(constants.saved_path+f'pred_epoch_{num_epoch}.png')
    plt.close()