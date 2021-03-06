from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import constants.constants as constants
from models.speech_to_speech.model import AutoEncoder
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistLossEpoch
from torch_dataset import TestSet, TrainSet


def test_loss(ae, testloader, criterion):
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        print("Calculate test loss...")
        total_loss = 0
        for iteration ,data in enumerate(testloader,0):
            input = data[0].to(device)
            target = input
            input = torch.reshape(input, (-1, input.shape[2], input.shape[1]))
            target = torch.reshape(target, (-1, target.shape[2], target.shape[1]))

            output = ae(input.float())
            loss = criterion(output, target.float())
            total_loss += loss.item()

        total_loss = total_loss/(iteration + 1)
        return total_loss


def train_model_speech_to_speech():
    print("Launching of model speech to speech")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batchsize = constants.batch_size
    n_epochs = constants.n_epochs

    trainset = TrainSet()
    trainset.scaling(True)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batchsize,shuffle=True)
    n_iteration_per_epoch = len(trainloader)

    testset = TestSet()
    testset.scaling(trainset.x_scaler, trainset.y_scaler)
    testloader = torch.utils.data.DataLoader(testset,batch_size=batchsize,shuffle=True)
    
    print("Saving params...")
    ae = AutoEncoder().to(device)
    optimizer = optim.Adam(ae.parameters(),lr=constants.g_lr)
    criterion = nn.MSELoss()
    save_params(constants.saved_path, ae)
    
    loss_tab = []
    t_loss_tab = []
    print("Starting Training Loop...")
    for epoch in range(0, n_epochs):
        start_epoch = datetime.now()
        current_loss = 0
        for iteration, data in enumerate(trainloader,0):
            print("*"+f"Starting iteration {iteration + 1}/{n_iteration_per_epoch}...")
            input = data[0].to(device)
            target = input
            input = torch.reshape(input, (-1, input.shape[2], input.shape[1]))
            target = torch.reshape(target, (-1, target.shape[2], target.shape[1]))
            ae.zero_grad()

            output = ae(input.float())
            loss = criterion(output, target.float())

            loss.backward() #gradients are computed
            optimizer.step()  #updates the parameters, the function can be called once the gradients are computed using e.g. backward().

            current_loss += loss.item() 
        
        current_loss = current_loss/(iteration + 1) #loss par epoch()
        loss_tab.append(current_loss)

        t_loss = test_loss(ae, testloader, criterion)
        t_loss_tab.append(t_loss)

        print ('[ %d ] loss : %.4f - t_loss : %.4f'% (epoch+1, current_loss, t_loss))

        if epoch % constants.log_interval == 0 or epoch >= n_epochs - 1:
            print("saving...")
            plotHistLossEpoch(epoch, loss_tab, t_loss_tab)
            saveModel(ae, epoch, constants.saved_path)
        
        end_epoch = datetime.now()   
        diff = end_epoch - start_epoch
        print("Duration of epoch :" + str(diff.total_seconds()))