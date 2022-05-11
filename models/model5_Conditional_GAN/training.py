from datetime import datetime

import numpy as np
from models.model5_Conditional_GAN.model import Generator, Discriminator
from torch_dataset import TestSet, TrainSet
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistLossEpochGAN, plotHistPredEpochGAN
import constant
import torch.nn as nn
import torch
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')
sns.set_style('whitegrid')

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('./runs/autoencoder')
#to visualize
#tensorboard --logdir ./runs/autoencoder/ 
def sample_noise(batch_size, dim):
    return np.random.normal(0, 1, (batch_size, dim))

def test_loss(G, D, testloader, criterion_loss):
    total_loss = 0
    for iteration ,data in enumerate(testloader,0):
        input, target = data
        input, target = Variable(input), Variable(target)
        input = torch.reshape(input, (-1, input.shape[2], input.shape[1]))
        target = torch.reshape(target, (-1, target.shape[2], target.shape[1]))

        real_batch_size = input.shape[0]
        noise = torch.Tensor(sample_noise(real_batch_size, constant.noise_size)).unsqueeze(1).to(device)

        output = G(input.float(), noise)
        gen_logit = D(output, input.float())
        gen_lable = torch.ones_like(gen_logit)

        adversarial_loss = criterion_loss(gen_logit, gen_lable)
        total_loss += adversarial_loss.data

    total_loss = total_loss/(iteration + 1)
    return total_loss.cpu().detach().numpy()


def train_model_5():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Training params
    n_epochs =  constant.n_epochs
    batch_size = constant.batch_size
    d_lr =  constant.d_lr  # Generator learning rate 
    g_lr =  constant.g_lr # Discriminator learning rate
    unroll_steps =  constant.unroll_steps
    log_interval =  constant.log_interval

    # --- Init dataset
    trainset = TrainSet()
    trainset.scaling(True)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)
    n_iteration_per_epoch = len(trainloader)

    testset = TestSet()
    testset.scaling(trainset.x_scaler, trainset.y_scaler)
    testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=2)

    G = Generator().to(device)
    D = Discriminator().to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=g_lr)
    d_opt = torch.optim.Adam(D.parameters(), lr=d_lr)

    print("Saving params...")
    save_params(constant.saved_path, G, D)
    
    bce_loss = torch.nn.BCELoss()

    d_loss_tab = []
    d_real_pred_tab = []
    d_fake_pred_tab = []
    g_loss_tab = []
    t_loss_tab = []

    print("Starting Training Loop...")
    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch + 1}/{n_epochs}...")
        start_epoch = datetime.now()
        current_d_loss = 0
        current_g_loss = 0
        current_fake_pred = 0
        current_real_pred = 0
        for iteration, data in enumerate(trainloader, 0):
            print("*"+f"Starting iteration {iteration + 1}/{n_iteration_per_epoch}...")
            torch.cuda.empty_cache()
            # * Configure real data
            inputs, targets = data
            inputs, targets = Variable(inputs), Variable(targets)
            inputs = torch.reshape(inputs, (-1, inputs.shape[2], inputs.shape[1]))
            targets = torch.reshape(targets, (-1, targets.shape[2], targets.shape[1]))
            real_batch_size = inputs.shape[0]
            # * Generate fake data
            print("** Generate fake data")
            noise = torch.Tensor(sample_noise(real_batch_size, constant.noise_size)).unsqueeze(1).to(device)
            with torch.no_grad():
                fake_targets = G(inputs.float(), noise) #le générateur génére les fausses données conditionnellement à la prosodie
            print("fake targer shape ", fake_targets.shape)

            # * Train D :  maximize log(D(x)) + log(1 - D(G(z)))
            print("** Train the discriminator")
            d_opt.zero_grad()
            real_logit = D(targets.float(), inputs.float()) #produce a result for each frame (tensor of length 300)
            fake_logit = D(fake_targets, inputs.float())

            #discriminator prediction
            current_real_pred += torch.mean(real_logit) #moy because the discriminator made a prediction for each frame
            current_fake_pred += torch.mean(fake_logit)

            real_label = torch.ones_like(real_logit) #tensor fill of 1 with the same size as input
            d_real_error = bce_loss(real_logit, real_label) #measures the Binary Cross Entropy between the target and the input probabilities

            fake_label = torch.zeros_like(fake_logit)
            d_fake_error = bce_loss(fake_logit, fake_label)
            
            d_loss = d_real_error + d_fake_error
            d_loss.backward() #gradients are computed
            d_opt.step() #updates the parameters, the function can be called once the gradients are computed using e.g. backward().

            current_d_loss += d_loss

            if unroll_steps:
                print("** Unroll D to reduce mode collapse")
                # * Unroll D to reduce mode collapse
                d_backup = D.state_dict() #a Python dictionary object that maps each layer to its parameter tensor.
                for _ in range(unroll_steps):
                    # * Train D
                    d_opt.zero_grad()

                    real_logit = D(targets.float(), inputs.float())
                    fake_logit = D(fake_targets, inputs.float())

                    real_label = torch.ones_like(real_logit)
                    d_real_error = bce_loss(real_logit, real_label)

                    fake_label = torch.zeros_like(fake_logit)
                    d_fake_error = bce_loss(fake_logit, fake_label)

                    d_loss = d_real_error + d_fake_error
                    d_loss.backward()
                    d_opt.step()

            # * Train G
            print("** Train the generator")
            g_opt.zero_grad()
            noise = torch.Tensor(sample_noise(real_batch_size, constant.noise_size)).unsqueeze(1).to(device)
            gen_y = G(inputs.float(), noise)
            gen_logit = D(gen_y, inputs.float())
            gen_lable = torch.ones_like(gen_logit)

            g_loss = bce_loss(gen_logit, gen_lable)
            g_loss.backward()
            g_opt.step()

            if unroll_steps:
                D.load_state_dict(d_backup)

            current_g_loss += g_loss

        #d_loss
        current_d_loss = current_d_loss/(iteration + 1) #loss par epoch
        d_loss_tab.append(current_d_loss.cpu().detach().numpy())
        
        #real pred
        current_real_pred = current_real_pred/(iteration + 1) 
        d_real_pred_tab.append(current_real_pred.cpu().detach().numpy())
        
        #fake pred
        current_fake_pred = current_fake_pred/(iteration + 1)
        d_fake_pred_tab.append(current_fake_pred.cpu().detach().numpy())

        #g_loss
        current_g_loss = current_g_loss/(iteration + 1)
        g_loss_tab.append(current_g_loss.cpu().detach().numpy())

        #test loss
        t_loss = test_loss(G, D, testloader, bce_loss)
        t_loss_tab.append(t_loss)
        if epoch % log_interval == 0 or epoch >= n_epochs - 1:
            print("saving...")
            saveModel(G, epoch, constant.saved_path)
            plotHistLossEpochGAN(epoch, d_loss_tab, g_loss_tab, t_loss_tab)
            plotHistPredEpochGAN(epoch, d_real_pred_tab, d_fake_pred_tab)

        end_epoch = datetime.now()   
        diff = end_epoch - start_epoch
        print("Duration of epoch :" + str(diff.total_seconds()))