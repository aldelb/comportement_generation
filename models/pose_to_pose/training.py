from datetime import datetime
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import constants.constants as constants
from models.TrainClass import Train
from models.pose_to_pose.model import AutoEncoder
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistAllLossEpoch, plotHistLossEpoch

class TrainModel7(Train):

    def __init__(self, gan):
        super(TrainModel7, self).__init__(gan)

    def test_loss(self, ae, testloader, criterion):
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Calculate test loss...")
            total_loss = 0
            for iteration ,data in enumerate(testloader,0):
                target = data[1].to(self.device)
                input = target
                input = torch.reshape(input, (-1, input.shape[2], input.shape[1]))
                target = torch.reshape(target, (-1, target.shape[2], target.shape[1]))

                output = ae(input.float())
                loss = criterion(output, target.float())
                total_loss += loss.item()

            total_loss = total_loss/(iteration + 1)
            return total_loss


    def train_model(self):
        print("Launching of model pose to pose")
        print("Saving params...")
        ae = AutoEncoder().to(self.device)
        optimizer = optim.Adam(ae.parameters(),lr=constants.g_lr)
        criterion = nn.MSELoss()
        save_params(constants.saved_path, ae)
        
        print("Starting Training Loop...")
        for epoch in range(0, self.n_epochs):
            start_epoch = datetime.now()
            self.reinitialize_loss()
            for iteration, data in enumerate(self.trainloader,0):
                print("*"+f"Starting iteration {iteration + 1}/{self.n_iteration_per_epoch}...")
                target = data[1].to(self.device)
                input = target
                input, target_eye, target_pose_r, target_au = self.format_data(input, target)
                target = torch.reshape(target, (-1, target.shape[2], target.shape[1])).float()

                ae.zero_grad()

                output = ae(input) 
                output_eye, output_pose_r, output_au = self.separate_openface_features(output)

                loss_eye = criterion(output_eye, target_eye)
                loss_pose_r = criterion(output_pose_r, target_pose_r)
                loss_au = criterion(output_au, target_au)
                loss = criterion(output, target)

                loss.backward() #gradients are computed
                optimizer.step()  #updates the parameters, the function can be called once the gradients are computed using e.g. backward().

                self.current_loss_eye += loss_eye.item()
                self.current_loss_pose_r += loss_pose_r.item()
                self.current_loss_au += loss_au.item()

                self.current_loss += loss .item()
            
            self.t_loss = self.test_loss(ae, self.testloader, criterion)
            self.update_loss_tab(iteration)

            print ('[ %d ] loss : %.4f - t_loss : %.4f'% (epoch+1, self.current_loss, self.t_loss))

            if epoch % constants.log_interval == 0 or epoch >= self.n_epochs - 1:
                print("saving...")
                plotHistLossEpoch(epoch, self.loss_tab, self.t_loss_tab)
                plotHistAllLossEpoch(epoch, self.loss_tab_eye, self.loss_tab_pose_r, self.loss_tab_au)
                saveModel(ae, epoch, constants.saved_path)
            
            end_epoch = datetime.now()   
            diff = end_epoch - start_epoch
            print("Duration of epoch :" + str(diff.total_seconds()))