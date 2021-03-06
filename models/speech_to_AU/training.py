from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import constants.constants as constants
from models.TrainClass import Train
from models.speech_to_AU.model import AutoEncoder
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistAllLossEpoch, plotHistLossEpoch

class TrainModel12(Train):

    def __init__(self, gan):
        super(TrainModel12, self).__init__(gan)

    def test_loss(self, ae, testloader, criterion_pose, criterion_au):
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Calculate test loss...")
            total_loss = 0
            for iteration, data in enumerate(testloader, 0):
                input, target = data[0].to(self.device), data[1].to(self.device)
                input, target_eye, target_pose_r, target_au = self.format_data(input, target)
                output_au = ae(input)

                loss_au = criterion_au(output_au, target_au.float())

                loss = loss_au
                total_loss += loss.item()

            total_loss = total_loss/(iteration + 1)
            return total_loss


    def train_model(self):
        print("Launching of model 10 : speech to head with autoencoder")
        print("Saving params...")
        ae = AutoEncoder().to(self.device)
        optimizer = optim.Adam(ae.parameters(), lr=constants.g_lr)
        criterionL2 = nn.MSELoss()
        save_params(constants.saved_path, ae)

        print("Starting Training Loop...")
        for epoch in range(0, self.n_epochs):
            start_epoch = datetime.now()
            self.reinitialize_loss()
            print(f"\nStarting epoch {epoch + 1}/{self.n_epochs}...")
            for iteration, data in enumerate(self.trainloader, 0):
                print("*"+f"Starting iteration {iteration + 1}/{self.n_iteration_per_epoch}...")
                torch.cuda.empty_cache()
                input, target = data[0].to(self.device), data[1].to(self.device)
                input, target_eye, target_pose_r, target_au = self.format_data(input, target)

                ae.zero_grad()

                output_au = ae(input)

                loss_au = criterionL2(output_au, target_au)

                loss = loss_au

                loss.backward()  # gradients are computed
                optimizer.step() # updates the parameters, the function can be called once the gradients are computed using e.g. backward().

                self.current_loss += loss.item()
    

            self.current_loss = self.current_loss/(iteration + 1)
            self.loss_tab.append(self.current_loss)

            self.t_loss = self.test_loss(ae, self.testloader, criterionL2, criterionL2)
            self.t_loss_tab.append(self.t_loss)


            print('[ %d ] loss : %.4f %.4f' % (epoch+1, self.current_loss, self.t_loss))

            if epoch % constants.log_interval == 0 or epoch >= self.n_epochs - 1:
                print("saving...")
                plotHistLossEpoch(epoch, self.loss_tab, self.t_loss_tab)
                saveModel(ae, epoch, constants.saved_path)

            end_epoch = datetime.now()
            diff = end_epoch - start_epoch
            print("Duration of epoch :" + str(diff.total_seconds()))
