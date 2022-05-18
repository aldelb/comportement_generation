from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import constant
from models.TrainClass import Train
from models.model2_skip_connectivity.model import AutoEncoder
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistAllLossEpoch, plotHistLossEpoch

class TrainModel2(Train):

    def __init__(self, gan):
        super(TrainModel2, self).__init__(gan)

    def test_loss(self, ae, testloader, criterion):
        print("Calculate test loss...")
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


    def train_model(self):
        print("Launching of model 2 : simple auto encoder with skip connectivity")
        print("Saving params...")
        ae = AutoEncoder()
        optimizer = optim.Adam(ae.parameters(),lr=constant.g_lr)
        criterion = nn.MSELoss()
        save_params(constant.saved_path, ae)

        print("Starting Training Loop...")
        for epoch in range(0, self.n_epochs):
            start_epoch = datetime.now()
            self.reinitialize_loss()
            for iteration, data in enumerate(self.trainloader,0):
                print("*"+f"Starting iteration {iteration + 1}/{self.n_iteration_per_epoch}...")
                input, target = data
                input, target_eye, target_pose_t, target_pose_r, target_au = self.format_data(input, target)
                target = Variable(target)
                target = torch.reshape(target, (-1, target.shape[2], target.shape[1]))

                optimizer.zero_grad()

                output = ae(input.float())
                output_eye, output_pose_t, output_pose_r, output_au = self.format_openface_output(output, dim=1)

                loss_eye = criterion(output_eye, target_eye.float())
                loss_pose_t = criterion(output_pose_t, target_pose_t.float())
                loss_pose_r = criterion(output_pose_r, target_pose_r.float())
                loss_au = criterion(output_au, target_au.float())
                loss = criterion(output, target.float())

                loss.backward() #gradients are computed
                optimizer.step()  #updates the parameters, the function can be called once the gradients are computed using e.g. backward().
                
                print(loss_eye)
                print(loss_pose_t)
                print(loss_pose_r)
                print(loss_au)
                print("loss ", loss)
                
                self.current_loss_eye += loss_eye
                self.current_loss_pose_t += loss_pose_t
                self.current_loss_pose_r += loss_pose_r
                self.current_loss_au += loss_au

                self.current_loss += loss 
            
            self.t_loss = self.test_loss(ae, self.testloader, criterion)
            self.update_loss_tab(iteration)

            print ('[ %d ] loss : %.4f - t_loss : %.4f'% (epoch+1, self.current_loss, self.t_loss))

            if epoch % constant.log_interval == 0 or epoch >= self.n_epochs - 1:
                print("saving...")
                plotHistLossEpoch(epoch, self.loss_tab, self.t_loss_tab)
                plotHistAllLossEpoch(epoch, self.loss_tab_eye, self.loss_tab_pose_t, self.loss_tab_pose_r, self.loss_tab_au, self.loss_tab)
                saveModel(ae, epoch, constant.saved_path)
            
            end_epoch = datetime.now()   
            diff = end_epoch - start_epoch
            print("Duration of epoch :" + str(diff.total_seconds()))