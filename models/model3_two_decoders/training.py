from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import constant
from models.model3_two_decoders.model import AutoEncoder
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistLossEpoch
from torch_dataset import TestSet, TrainSet
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('./runs/autoencoder')
#to visualize
#tensorboard --logdir ./runs/autoencoder/ 

def test_loss(ae, testloader, criterion_pose, criterion_au):
    total_loss = 0
    for iteration ,data in enumerate(testloader,0):
        input, target = data
        target_pose = torch.index_select(target, 2, torch.tensor(range(constant.pose_size)))
        target_au = torch.index_select(target, 2, torch.tensor(range(constant.pose_size, constant.pose_size + constant.au_size)))

        input, target_pose, target_au = Variable(input), Variable(target_pose),  Variable(target_au)
        input = torch.reshape(input, (-1, input.shape[2], input.shape[1]))
        target_pose = torch.reshape(target_pose, (-1, target_pose.shape[2], target_pose.shape[1]))
        target_au = torch.reshape(target_au, (-1, target_au.shape[2], target_au.shape[1]))

        output_pose, output_au = ae(input.float())

        loss_pose = criterion_pose(output_pose, target_pose.float())
        loss_au = criterion_au(output_au, target_au.float())

        loss = loss_pose + loss_au 
        total_loss += loss.data

    total_loss = total_loss/(iteration + 1)
    return total_loss.cpu().detach().numpy()


def train_model_3():
    print("Launching of model 3 : auto encoder with two decoders")
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
    criterionL2 = nn.MSELoss()
    criterionL1 = nn.L1Loss()
    save_params(constant.saved_path, ae)

    ####Tensorboard visualisation#########
    # print("TensorBoard visualisation...")
    # dataiter = iter(trainloader)
    # prosodie, pose = dataiter.next()
    # prosodie =  Variable(prosodie)
    # prosodie = torch.reshape(prosodie, (-1, prosodie.shape[2], prosodie.shape[1]))
    # writer.add_graph(ae, prosodie.float())
    # writer.close()
    #######################################
    
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
            target_pose = torch.index_select(target, 2, torch.tensor(range(constant.pose_size)))
            target_au = torch.index_select(target, 2, torch.tensor(range(constant.pose_size, constant.pose_size + constant.au_size)))

            input, target_pose, target_au = Variable(input), Variable(target_pose),  Variable(target_au)
            input = torch.reshape(input, (-1, input.shape[2], input.shape[1]))
            target_pose = torch.reshape(target_pose, (-1, target_pose.shape[2], target_pose.shape[1]))
            target_au = torch.reshape(target_au, (-1, target_au.shape[2], target_au.shape[1]))
            optimizer.zero_grad()

            output_pose, output_au = ae(input.float())

            loss_pose = criterionL2(output_pose, target_pose.float())
            loss_au = criterionL2(output_au, target_au.float())

            loss = loss_pose + loss_au ##add weight ??

            loss.backward() #gradients are computed
            optimizer.step()  #updates the parameters, the function can be called once the gradients are computed using e.g. backward().

            current_loss += loss 
        
        current_loss = current_loss/(iteration + 1) #loss par epoch
        current_loss = current_loss.cpu().detach().numpy()
        loss_tab.append(current_loss)

        t_loss = test_loss(ae, testloader, criterionL2, criterionL2)
        t_loss_tab.append(t_loss)

        print ('[ %d ] loss : %.4f %.4f'% (epoch+1, current_loss, t_loss))

        if epoch % constant.log_interval == 0 or epoch >= n_epochs - 1:
            print("saving...")
            plotHistLossEpoch(epoch, loss_tab, t_loss_tab)
            saveModel(ae, epoch, constant.saved_path)

        end_epoch = datetime.now()   
        diff = end_epoch - start_epoch
        print("Duration of epoch :" + str(diff.total_seconds()))