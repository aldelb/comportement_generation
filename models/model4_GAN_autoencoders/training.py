from datetime import datetime
from models.model4_GAN_autoencoders.model import Generator, Discriminator
from torch_dataset import TrainSet
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

def train_model_4():
    print("Launching of model 4 : GAN with auto encoder as generator")

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

    G = Generator().to(device)
    D = Discriminator().to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=g_lr)
    d_opt = torch.optim.Adam(D.parameters(), lr=d_lr)

    print("Saving params...")
    save_params(constant.saved_path, G, D)


    ####Tensorboard visualisation#########
    # print("TensorBoard visualisation...")
    # dataiter = iter(trainloader)
    # prosodie, pose = dataiter.next()
    # prosodie, pose =  Variable(prosodie), Variable(pose)
    # prosodie, pose = torch.reshape(prosodie, (-1, prosodie.shape[2], prosodie.shape[1])), torch.reshape(pose, (-1, pose.shape[2], pose.shape[1]))
    # #writer.add_graph(G, prosodie.float())
    # writer.add_graph(D, (prosodie.float(), pose.float()))
    # writer.close()
    #######################################

    bce_loss = torch.nn.BCELoss()
    criterionL2 = nn.MSELoss()
    criterionL1 = nn.L1Loss()

    d_loss_tab = []
    d_real_pred = []
    d_fake_pred = []
    g_loss_tab = []

    print("Starting Training Loop...")
    for epoch in range(n_epochs):
        start_epoch = datetime.now()
        print(f"\nStarting epoch {epoch + 1}/{n_epochs}...")
        for iteration, data in enumerate(trainloader, 0):
            print("*"+f"Starting iteration {iteration + 1}/{n_iteration_per_epoch}...")
            torch.cuda.empty_cache()
            # * Configure real data
            inputs, targets = data
            targets_pose = torch.index_select(targets, 2, torch.tensor(range(constant.pose_size)))
            targets_au = torch.index_select(targets, 2, torch.tensor(range(constant.pose_size, constant.pose_size + constant.au_size)))

            inputs, targets, targets_pose, targets_au = Variable(inputs), Variable(targets), Variable(targets_pose), Variable(targets_au)
            inputs = torch.reshape(inputs, (-1, inputs.shape[2], inputs.shape[1]))
            targets = torch.reshape(targets, (-1, targets.shape[2], targets.shape[1]))
            targets_pose = torch.reshape(targets_pose, (-1, targets_pose.shape[2], targets_pose.shape[1]))
            targets_au = torch.reshape(targets_au, (-1, targets_au.shape[2], targets_au.shape[1]))

            # * Generate fake data
            with torch.no_grad():
                output_pose, output_au = G(inputs.float()) #le générateur génére les fausses données conditionnellement à la prosodie
                fake_targets = torch.cat((output_pose, output_au), 1)

            # * Train D :  maximize log(D(x)) + log(1 - D(G(z)))
            print("** Train the discriminator")
            d_opt.zero_grad()
            real_logit = D(targets.float(), inputs.float()) #produce a result for each frame (tensor of length 300)
            fake_logit = D(fake_targets, inputs.float())

            d_real_pred_moy = torch.mean(real_logit) #moy because the discriminator made a prediction for each frame
            d_fake_pred_moy = torch.mean(fake_logit)

            real_label = torch.ones_like(real_logit) #tensor fill of 1 with the same size as input
            d_real_error = bce_loss(real_logit, real_label) #measures the Binary Cross Entropy between the target and the input probabilities

            fake_label = torch.zeros_like(fake_logit)
            d_fake_error = bce_loss(fake_logit, fake_label)
            
            d_loss = d_real_error + d_fake_error
            d_loss.backward() #gradients are computed
            d_opt.step() #updates the parameters, the function can be called once the gradients are computed using e.g. backward().

            d_loss_tab.append(d_loss.cpu().detach().numpy())
            d_fake_pred.append(d_fake_pred_moy.cpu().detach().numpy())
            d_real_pred.append(d_real_pred_moy.cpu().detach().numpy())


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
            gen_pose, gen_au = G(inputs.float())
            gen_y = torch.cat((gen_pose, gen_au), 1)
            gen_logit = D(gen_y, inputs.float())
            gen_lable = torch.ones_like(gen_logit)

            adversarial_loss = bce_loss(gen_logit, gen_lable)
            loss_pose = criterionL2(gen_pose, targets_pose.float())
            loss_au = criterionL2(gen_au, targets_au.float())

            g_loss = loss_pose + loss_au + adversarial_loss
            g_loss.backward()
            g_opt.step()

            if unroll_steps:
                D.load_state_dict(d_backup)

            g_loss_tab.append(g_loss.cpu().detach().numpy())

        if epoch % log_interval == 0 or epoch >= n_epochs - 1:
            print("saving...")
            saveModel(G, epoch, constant.saved_path)
            plotHistLossEpochGAN(epoch, n_iteration_per_epoch, d_loss_tab, g_loss_tab, constant.saved_path)
            plotHistPredEpochGAN(epoch, n_iteration_per_epoch, d_real_pred, d_fake_pred, constant.saved_path)
            print("end of save")

        end_epoch = datetime.now()   
        diff = end_epoch - start_epoch
        print("Duration of epoch :" + str(diff.total_seconds()))
