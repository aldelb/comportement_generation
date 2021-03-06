from datetime import datetime
import torch
import torch.nn as nn
import constants.constants as constants
from models.TrainClass import Train
from models.speech_to_head_GAN.model import Generator, Discriminator
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistLossEpochGAN, plotHistPredEpochGAN

class TrainModel11(Train):

    def __init__(self, gan):
        super(TrainModel11, self).__init__(gan)

    def test_loss(self, G, D, testloader, criterion_pose, criterion_adv):
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Calculate test loss...")
            total_loss = 0
            for iteration, data in enumerate(testloader, 0):
                input, target = data[0].to(self.device), data[1].to(self.device)
                input, target_eye, target_pose_r, target_au = self.format_data(input, target)
                
                gen_pose_r = G(input)
                gen_logit = D(gen_pose_r, input)
                gen_lable = torch.ones_like(gen_logit)

                loss_pose_r = criterion_pose(gen_pose_r, target_pose_r.float())
                adversarial_loss = (constants.adversarial_coeff * criterion_adv(gen_logit, gen_lable))

                loss = loss_pose_r + adversarial_loss
                total_loss += loss.item()

            total_loss = total_loss/(iteration + 1)
            return total_loss


    def train_model(self):
        print("Launching of model 11 : speech to head with autoencoder GAN")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        G = Generator().to(device)
        D = Discriminator().to(device)

        g_opt = torch.optim.Adam(G.parameters(), lr=constants.g_lr)
        d_opt = torch.optim.Adam(D.parameters(), lr=constants.d_lr)

        print("Saving params...")
        save_params(constants.saved_path, G, D)

        bce_loss = torch.nn.BCELoss()
        criterionL2 = nn.MSELoss()

        print("Starting Training Loop...")
        for epoch in range(0, self.n_epochs):
            start_epoch = datetime.now()
            self.reinitialize_loss()
            print(f"\nStarting epoch {epoch + 1}/{self.n_epochs}...")
            for iteration, data in enumerate(self.trainloader, 0):
                torch.cuda.empty_cache()
                print("*"+f"Starting iteration {iteration + 1}/{self.n_iteration_per_epoch}...")
                #configure real data
                input, target = data[0].to(self.device), data[1].to(self.device)
                input, target_eye, target_pose_r, target_au = self.format_data(input, target)
                target = target_pose_r

                # * Generate fake data
                with torch.no_grad():
                    output_pose_r = G(input) #le g??n??rateur g??n??re les fausses donn??es conditionnellement ?? la prosodie
                    fake_target = output_pose_r

                # * Train D :  maximize log(D(x)) + log(1 - D(G(z)))
                print("** Train the discriminator")
                D.zero_grad()
                real_logit = D(target, input) #produce a result for each frame (tensor of length 300)
                fake_logit = D(fake_target, input)
                
                #discriminator prediction
                self.current_real_pred += torch.mean(real_logit).item() #moy because the discriminator made a prediction for each frame
                self.current_fake_pred += torch.mean(fake_logit).item()


                #discriminator loss
                real_label = torch.ones_like(real_logit) #tensor fill of 1 with the same size as input
                d_real_error = bce_loss(real_logit, real_label) #measures the Binary Cross Entropy between the target and the input probabilities
                d_real_error.backward()

                fake_label = torch.zeros_like(fake_logit)
                d_fake_error = bce_loss(fake_logit, fake_label)
                d_fake_error.backward()

                d_loss = d_real_error + d_fake_error
                d_opt.step() #updates the parameters, the function can be called once the gradients are computed using e.g. backward().

                self.current_d_loss += d_loss.item()

                if constants.unroll_steps:
                    print("** Unroll D to reduce mode collapse")
                    # * Unroll D to reduce mode collapse
                    d_backup = D.state_dict() #a Python dictionary object that maps each layer to its parameter tensor.
                    for _ in range(constants.unroll_steps):
                        # * Train D
                        D.zero_grad()

                        real_logit = D(target, input)
                        fake_logit = D(fake_target, input)

                        real_label = torch.ones_like(real_logit)
                        d_real_error = bce_loss(real_logit, real_label)
                        d_real_error.backward()

                        fake_label = torch.zeros_like(fake_logit)
                        d_fake_error = bce_loss(fake_logit, fake_label)
                        d_fake_error.backward()

                        d_loss = d_real_error + d_fake_error
                        d_opt.step()

                # * Train G
                print("** Train the generator")
                G.zero_grad()
                gen_pose_r = G(input)
                gen_logit = D(gen_pose_r, input)
                gen_lable = torch.ones_like(gen_logit)

                adversarial_loss = (constants.adversarial_coeff * bce_loss(gen_logit, gen_lable))
                loss_pose_r = criterionL2(gen_pose_r, target_pose_r)

                g_loss = loss_pose_r + adversarial_loss
                g_loss.backward()
                g_opt.step()

                if constants.unroll_steps:
                    D.load_state_dict(d_backup)

                self.current_loss += g_loss.item()
                
                
            self.current_loss = self.current_loss/(iteration + 1)
            self.loss_tab.append(self.current_loss)

            self.t_loss = self.test_loss(G, D, self.testloader, criterionL2, bce_loss)
            self.t_loss_tab.append(self.t_loss)

            #d_loss
            self.current_d_loss = self.current_d_loss/(iteration + 1) #loss par epoch
            self.d_loss_tab.append(self.current_d_loss)
            
            #real pred
            self.current_real_pred = self.current_real_pred/(iteration + 1) 
            self.d_real_pred_tab.append(self.current_real_pred)
            
            #fake pred
            self.current_fake_pred = self.current_fake_pred/(iteration + 1)
            self.d_fake_pred_tab.append(self.current_fake_pred)

            print('[ %d ] g_loss : %.4f, t_loss : %.4f' % (epoch+1, self.current_loss, self.t_loss))
            print('[ %d ] head_loss : %.4f, adv_loss : %.4f' % (epoch+1, loss_pose_r, adversarial_loss))

            if epoch % constants.log_interval == 0 or epoch >= self.n_epochs - 1:
                print("saving...")
                plotHistLossEpochGAN(epoch, self.d_loss_tab, self.loss_tab, self.t_loss_tab)
                plotHistPredEpochGAN(epoch, self.d_real_pred_tab, self.d_fake_pred_tab)                
                saveModel(G, epoch, constants.saved_path)


            end_epoch = datetime.now()
            diff = end_epoch - start_epoch
            print("Duration of epoch :" + str(diff.total_seconds()))
