import torch
import constants.constants as constants
from torch_dataset import TrainSet

class Generate():
    def __init__(self):
        self.dset = TrainSet()

    def reshape_input(self, inputs):
        inputs = self.dset.scale_x(inputs)
        inputs = torch.FloatTensor(inputs).unsqueeze(1)
        inputs = torch.reshape(inputs, (-1, inputs.shape[2], inputs.shape[0]))
        return inputs
    
    def reshape_input_pose(self, inputs):
        inputs = self.dset.scale_y(inputs)
        inputs = torch.FloatTensor(inputs).unsqueeze(1)
        inputs = torch.reshape(inputs, (-1, inputs.shape[2], inputs.shape[0]))
        return inputs

    def separate_openface_features(self, input, dim):
        input_eye = torch.index_select(input, dim, torch.tensor(range(constants.eye_size)))
        input_pose_r = torch.index_select(input, dim, torch.tensor(range(constants.eye_size, constants.eye_size + constants.pose_r_size)))
        input_au = torch.index_select(input, dim, torch.tensor(range(constants.pose_size, constants.pose_size + constants.au_size)))

        return input_eye, input_pose_r, input_au
    

    def reshape_output(self, output_eye, output_pose_r, output_au):
        outs = torch.cat((output_eye, output_pose_r, output_au), 1)
        outs = torch.FloatTensor(outs)
        outs = torch.reshape(outs, (-1, constants.pose_size + constants.au_size))
        outs = self.dset.rescale_y(outs)
        return outs
    
    def reshape_single_output(self, outs):
        outs = torch.FloatTensor(outs)
        outs = torch.reshape(outs, (-1, constants.pose_size + constants.au_size))
        outs = self.dset.rescale_y(outs)
        return outs