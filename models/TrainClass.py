import torch
import constant
from torch_dataset import TestSet, TrainSet
from torch.autograd import Variable

class Train():
    def __init__(self, gan=False):
        self.gan = gan
        self.batchsize = constant.batch_size
        self.n_epochs = constant.n_epochs

        trainset = TrainSet()
        trainset.scaling(True)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batchsize,shuffle=True,num_workers=2)
        self.n_iteration_per_epoch = len(self.trainloader)

        testset = TestSet()
        testset.scaling(trainset.x_scaler, trainset.y_scaler)
        self.testloader = torch.utils.data.DataLoader(testset,batch_size=self.batchsize,shuffle=True,num_workers=2)
        
        self.reinitialize_loss()
        self.reinitialize_loss_tab()

    def reinitialize_loss_tab(self):
        self.loss_tab_eye = []
        self.loss_tab_pose_t = []
        self.loss_tab_pose_r = []
        self.loss_tab_au = []
        self.loss_tab = []
        self.t_loss_tab = []

        if(self.gan):
            self.d_loss_tab = []
            self.d_real_pred_tab = []
            self.d_fake_pred_tab = []


    def update_loss_tab(self, iteration):
        self.current_loss_eye = self.current_loss_eye/(iteration + 1)
        self.loss_tab_eye.append(self.current_loss_eye.cpu().detach().numpy())

        self.current_loss_pose_t = self.current_loss_pose_t/(iteration + 1)
        self.loss_tab_pose_t.append(self.current_loss_pose_t.cpu().detach().numpy())

        self.current_loss_pose_r = self.current_loss_pose_r/(iteration + 1)
        self.loss_tab_pose_r.append(self.current_loss_pose_r.cpu().detach().numpy())

        self.current_loss_au = self.current_loss_au/(iteration + 1)
        self.loss_tab_au.append(self.current_loss_au.cpu().detach().numpy())

        self.current_loss = self.current_loss/(iteration + 1)  # loss par epoch
        self.loss_tab.append(self.current_loss.cpu().detach().numpy())

        self.t_loss_tab.append(self.t_loss)

        if(self.gan):
            #d_loss
            self.current_d_loss = self.current_d_loss/(iteration + 1) #loss par epoch
            self.d_loss_tab.append(self.current_d_loss.cpu().detach().numpy())
            
            #real pred
            self.current_real_pred = self.current_real_pred/(iteration + 1) 
            self.d_real_pred_tab.append(self.current_real_pred.cpu().detach().numpy())
            
            #fake pred
            self.current_fake_pred = self.current_fake_pred/(iteration + 1)
            self.d_fake_pred_tab.append(self.current_fake_pred.cpu().detach().numpy())


    def reinitialize_loss(self):
        self.current_loss_eye = 0
        self.current_loss_pose_t = 0
        self.current_loss_pose_r = 0
        self.current_loss_au = 0
        self.current_loss = 0
        self.t_loss = 0

        if(self.gan):
            self.current_d_loss = 0
            self.current_fake_pred = 0
            self.current_real_pred = 0

    def format_data(self, input, target):
        input, target = Variable(input), Variable(target)
        input = torch.reshape(input, (-1, input.shape[2], input.shape[1]))
        target = torch.reshape(target, (-1, target.shape[2], target.shape[1]))
        target_eye, target_pose_t, target_pose_r, target_au = self.separate_openface_features(target, dim=1)

        return input.float(), target_eye.float(), target_pose_t.float(), target_pose_r.float(), target_au.float()

    def separate_openface_features(self, output, dim):
        output_eye = torch.index_select(output, dim, torch.tensor(range(constant.eye_size)))
        output_pose_t = torch.index_select(output, dim, torch.tensor(range(constant.eye_size, constant.eye_size + constant.pose_t_size)))
        output_pose_r = torch.index_select(output, dim, torch.tensor(range(constant.eye_size + constant.pose_t_size, constant.eye_size + constant.pose_t_size + constant.pose_r_size)))
        output_au = torch.index_select(output, dim, torch.tensor(range(constant.pose_size, constant.pose_size + constant.au_size)))

        return output_eye, output_pose_t, output_pose_r, output_au
    