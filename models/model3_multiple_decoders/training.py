from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import constant
from models.TrainClass import Train
from models.model3_multiple_decoders.model import AutoEncoder
from utils.model_utils import saveModel
from utils.params_utils import save_params
from utils.plot_utils import plotHistAllLossEpoch, plotHistLossEpoch
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('./runs/autoencoder')
# to visualize
# tensorboard --logdir ./runs/autoencoder/

class TrainModel3(Train):

    def __init__(self, gan):
        super(TrainModel3, self).__init__(gan)

    def test_loss(self, ae, testloader, criterion_pose, criterion_au):
        total_loss = 0
        for iteration, data in enumerate(testloader, 0):
            input, target = data
            input, target_eye, target_pose_t, target_pose_r, target_au = self.format_data(input, target)
            output_eye, output_pose_t, output_pose_r, output_au = ae(input)

            loss_eye = criterion_pose(output_eye, target_eye.float())
            loss_pose_t = criterion_pose(output_pose_t, target_pose_t.float())
            loss_pose_r = criterion_pose(output_pose_r, target_pose_r.float())
            loss_au = criterion_au(output_au, target_au.float())

            loss = loss_eye + loss_pose_t + loss_pose_r + loss_au
            total_loss += loss.data

        total_loss = total_loss/(iteration + 1)
        return total_loss.cpu().detach().numpy()


    def train_model(self):
        print("Launching of model 3 : auto encoder with four decoders")
        print("Saving params...")
        ae = AutoEncoder()
        optimizer = optim.Adam(ae.parameters(), lr=constant.g_lr)
        criterionL2 = nn.MSELoss()
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

        print("Starting Training Loop...")
        for epoch in range(0, self.n_epochs):
            start_epoch = datetime.now()
            self.reinitialize_loss()
            print(f"\nStarting epoch {epoch + 1}/{self.n_epochs}...")
            for iteration, data in enumerate(self.trainloader, 0):
                print("*"+f"Starting iteration {iteration + 1}/{self.n_iteration_per_epoch}...")
                input, target = data
                input, target_eye, target_pose_t, target_pose_r, target_au = self.format_data(input, target)

                optimizer.zero_grad()

                output_eye, output_pose_t, output_pose_r, output_au = ae(input)

                loss_eye = criterionL2(output_eye, target_eye.float())
                loss_pose_t = criterionL2(output_pose_t, target_pose_t.float())
                loss_pose_r = criterionL2(output_pose_r, target_pose_r.float())
                loss_au = criterionL2(output_au, target_au.float())

                loss = loss_eye + loss_pose_t + loss_pose_r + loss_au  # add weight ??

                loss.backward()  # gradients are computed
                optimizer.step() # updates the parameters, the function can be called once the gradients are computed using e.g. backward().

                self.current_loss_eye += loss_eye
                self.current_loss_pose_t += loss_pose_t
                self.current_loss_pose_r += loss_pose_r
                self.current_loss_au += loss_au

                self.current_loss += loss
    
            self.t_loss = self.test_loss(ae, self.testloader, criterionL2, criterionL2)
            self.update_loss_tab(iteration)


            print('[ %d ] loss : %.4f %.4f' % (epoch+1, self.current_loss, self.t_loss))

            if epoch % constant.log_interval == 0 or epoch >= self.n_epochs - 1:
                print("saving...")
                plotHistLossEpoch(epoch, self.loss_tab, self.t_loss_tab)
                plotHistAllLossEpoch(epoch, self.loss_tab_eye, self.loss_tab_pose_t, self.loss_tab_pose_r, self.loss_tab_au, self.loss_tab)
                saveModel(ae, epoch, constant.saved_path)

            end_epoch = datetime.now()
            diff = end_epoch - start_epoch
            print("Duration of epoch :" + str(diff.total_seconds()))
