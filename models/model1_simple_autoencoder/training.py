from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import constant
from models.model1_simple_autoencoder.model import AutoEncoder
from torch_dataset import TestSet, TrainSet
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistLossEpoch

def test_loss(ae, testloader, criterion):
    total_loss = 0
    for iteration ,data in enumerate(testloader,0):
        input, target = data
        input, target = Variable(input), Variable(target)
        input = torch.reshape(input, (-1, input.shape[2], input.shape[1]))
        target = torch.reshape(target, (-1, target.shape[2], target.shape[1]))

        output = ae(input.float())
        loss = criterion(output, target.float())

        total_loss += loss.data

    total_loss = total_loss/(iteration + 1)
    return total_loss.cpu().detach().numpy()


def train_model_1():
    print("Launching of model 1 : simple auto encoder")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batchsize = constant.batch_size
    n_epochs = constant.n_epochs

    trainset = TrainSet()
    trainset.scaling(True)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batchsize,shuffle=True,num_workers=2)
    n_iteration_per_epoch = len(trainloader)

    testset = TestSet()
    testset.scaling(trainset.x_scaler, trainset.y_scaler)
    testloader = torch.utils.data.DataLoader(testset,batch_size=batchsize,shuffle=True,num_workers=2)
  
    print("Saving params...")
    ae = AutoEncoder()
    optimizer = optim.Adam(ae.parameters(),lr=constant.g_lr)
    criterion = nn.MSELoss()
    save_params(constant.saved_path, ae)
    
    
    loss_tab = []
    t_loss_tab = []
    print("Starting Training Loop...")
    for epoch in range(0, n_epochs):
        start_epoch = datetime.now()
        current_loss = 0
        print(f"\nStarting epoch {epoch + 1}/{n_epochs}...")
        for iteration, data in enumerate(trainloader,0):
            print("*"+f"Starting iteration {iteration + 1}/{n_iteration_per_epoch}...")
            input, target = data
            input, target = Variable(input), Variable(target)
            input = torch.reshape(input, (-1, input.shape[2], input.shape[1]))
            target = torch.reshape(target, (-1, target.shape[2], target.shape[1]))
            ae.zero_grad()

            output = ae(input.float())
            loss = criterion(output, target.float())

            loss.backward() #gradients are computed
            optimizer.step()  #updates the parameters, the function can be called once the gradients are computed using e.g. backward().

            current_loss += loss 
        
        current_loss = current_loss/(iteration + 1) #loss par epoch
        current_loss = current_loss.cpu().detach().numpy()
        loss_tab.append(current_loss)

        t_loss = test_loss(ae, testloader, criterion)
        t_loss_tab.append(t_loss)

        print ('[ %d ] loss : %.4f - t_loss : %.4f'% (epoch+1, current_loss, t_loss))

        if epoch % constant.log_interval == 0 or epoch >= n_epochs - 1:
            print("saving...")
            plotHistLossEpoch(epoch, loss_tab, t_loss_tab)
            saveModel(ae, epoch, constant.saved_path)
        
        end_epoch = datetime.now()   
        diff = end_epoch - start_epoch
        print("Duration of epoch :" + str(diff.total_seconds()))

