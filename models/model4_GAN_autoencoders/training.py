from datetime import datetime
from models.TrainClass import Train
from models.model4_GAN_autoencoders.model import Generator, Discriminator
from torch_dataset import TestSet, TrainSet
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistAllLossEpoch, plotHistLossEpochGAN, plotHistPredEpochGAN
import constants.constants as constants
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

class TrainModel4(Train):

    def __init__(self, gan):
       super(TrainModel4, self).__init__(gan)

    def test_loss(self, G, D, testloader, criterion_pose, criterion_au, criterion_loss):
        total_loss = 0
        for iteration ,data in enumerate(testloader,0):
            input, target = data
            input, target_eye, target_pose_t, target_pose_r, target_au = self.format_data(input, target)

            gen_eye, gen_pose_t, gen_pose_r, gen_au = G(input.float())
            gen_y = torch.cat((gen_eye, gen_pose_t, gen_pose_r, gen_au), 1)
            gen_logit = D(gen_y, input.float())
            gen_lable = torch.ones_like(gen_logit)
            
            loss_eye = criterion_pose(gen_eye, target_eye.float())
            loss_pose_t = criterion_pose(gen_pose_t, target_pose_t.float())
            loss_pose_r = criterion_pose(gen_pose_r, target_pose_r.float())
            loss_au = criterion_au(gen_au, target_au.float())

            adversarial_loss = criterion_loss(gen_logit, gen_lable)
            
            loss = loss_eye + loss_pose_t + loss_pose_r + loss_au + adversarial_loss
            total_loss += loss.data

        total_loss = total_loss/(iteration + 1)
        return total_loss.cpu().detach().numpy()

    def train_model(self):
        print("Launching of model 4 : GAN with auto encoder as generator")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        G = Generator().to(device)
        D = Discriminator().to(device)

        g_opt = torch.optim.Adam(G.parameters(), lr=constants.g_lr)
        d_opt = torch.optim.Adam(D.parameters(), lr=constants.d_lr)

        print("Saving params...")
        save_params(constants.saved_path, G, D)


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

        print("Starting Training Loop...")
        for epoch in range(self.n_epochs):
            print(f"Starting epoch {epoch + 1}/{self.n_epochs}...")
            start_epoch = datetime.now()
            self.reinitialize_loss()
            for iteration, data in enumerate(self.trainloader, 0):
                print("*"+f"Starting iteration {iteration + 1}/{self.n_iteration_per_epoch}...")
                torch.cuda.empty_cache()
                # * Configure real data
                inputs, targets = data
                inputs, target_eye, target_pose_t, target_pose_r, target_au = self.format_data(inputs, targets)
                targets = Variable(targets)
                targets = torch.reshape(targets, (-1, targets.shape[2], targets.shape[1])).float()

                # * Generate fake data
                with torch.no_grad():
                    output_eye, output_pose_t, output_pose_r, output_au = G(inputs) #le générateur génére les fausses données conditionnellement à la prosodie
                    fake_targets = torch.cat((output_eye, output_pose_t, output_pose_r, output_au), 1)

                # * Train D :  maximize log(D(x)) + log(1 - D(G(z)))
                print("** Train the discriminator")
                d_opt.zero_grad()
                real_logit = D(targets, inputs) #produce a result for each frame (tensor of length 300)
                fake_logit = D(fake_targets, inputs)

                #discriminator prediction
                self.current_real_pred += torch.mean(real_logit) #moy because the discriminator made a prediction for each frame
                self.current_fake_pred += torch.mean(fake_logit)

                #discriminator loss
                real_label = torch.ones_like(real_logit) #tensor fill of 1 with the same size as input
                d_real_error = bce_loss(real_logit, real_label) #measures the Binary Cross Entropy between the target and the input probabilities

                fake_label = torch.zeros_like(fake_logit)
                d_fake_error = bce_loss(fake_logit, fake_label)
                
                d_loss = d_real_error + d_fake_error
                d_loss.backward() #gradients are computed
                d_opt.step() #updates the parameters, the function can be called once the gradients are computed using e.g. backward().

                self.current_d_loss += d_loss

                if constants.unroll_steps:
                    print("** Unroll D to reduce mode collapse")
                    # * Unroll D to reduce mode collapse
                    d_backup = D.state_dict() #a Python dictionary object that maps each layer to its parameter tensor.
                    for _ in range(constants.unroll_steps):
                        # * Train D
                        d_opt.zero_grad()

                        real_logit = D(targets, inputs)
                        fake_logit = D(fake_targets, inputs)

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
                gen_eye, gen_pose_t, gen_pose_r, gen_au = G(inputs)
                gen_y = torch.cat((gen_eye, gen_pose_t, gen_pose_r, gen_au), 1)
                gen_logit = D(gen_y, inputs)
                gen_lable = torch.ones_like(gen_logit)

                adversarial_loss = bce_loss(gen_logit, gen_lable)
                loss_eye = criterionL2(gen_eye, target_eye)
                loss_pose_t = criterionL2(gen_pose_t, target_pose_t)
                loss_pose_r = criterionL2(gen_pose_r, target_pose_r)
                loss_au = criterionL2(gen_au, target_au)

                g_loss = loss_eye + loss_pose_t + loss_pose_r + loss_au + adversarial_loss
                g_loss.backward()
                g_opt.step()

                if constants.unroll_steps:
                    D.load_state_dict(d_backup)

                self.current_loss_eye += loss_eye
                self.current_loss_pose_t += loss_pose_t
                self.current_loss_pose_r += loss_pose_r
                self.current_loss_au += loss_au

                self.current_loss += g_loss

            self.t_loss = self.test_loss(G, D, self.testloader, criterionL2, criterionL2, bce_loss)
            self.update_loss_tab(iteration)

            print('[ %d ] loss : %.4f %.4f' % (epoch+1, self.current_loss, self.t_loss))

            if epoch % constants.log_interval == 0 or epoch >= self.n_epochs - 1:
                print("saving...")
                saveModel(G, epoch, constants.saved_path)
                plotHistLossEpochGAN(epoch, self.d_loss_tab, self.loss_tab, self.t_loss_tab)
                plotHistPredEpochGAN(epoch, self.d_real_pred_tab, self.d_fake_pred_tab)
                plotHistAllLossEpoch(epoch, self.loss_tab_eye, self.loss_tab_pose_t, self.loss_tab_pose_r, self.loss_tab_au, self.loss_tab)

            end_epoch = datetime.now()   
            diff = end_epoch - start_epoch
            print("Duration of epoch :" + str(diff.total_seconds()))
