import torch
import constant
from torch_dataset import TrainSet

class Generate():
    def __init__(self):
        self.dset = TrainSet()

    def reshape_input(self, inputs):
        inputs = self.dset.scale_x(inputs)
        inputs = torch.FloatTensor(inputs).unsqueeze(1)
        inputs = torch.reshape(inputs, (-1, inputs.shape[2], inputs.shape[0]))
        return inputs

    def reshape_output(self, output_eye, output_pose_t, output_pose_r, output_au):
        output_eye = torch.FloatTensor(output_eye)
        output_eye = torch.reshape(output_eye, (-1, constant.eye_size))

        output_pose_t = torch.FloatTensor(output_pose_t)
        output_pose_t = torch.reshape(output_pose_t, (-1, constant.pose_t_size))

        output_pose_r = torch.FloatTensor(output_pose_r)
        output_pose_r = torch.reshape(output_pose_r, (-1, constant.pose_r_size))

        output_au = torch.FloatTensor(output_au)
        output_au = torch.reshape(output_au, (-1, constant.au_size))
        
        outs = torch.cat((output_eye, output_pose_t, output_pose_r, output_au), 1)
        outs = self.dset.rescale_y(outs)

        return outs
    
    def reshape_single_output(self, outs):
        outs = torch.FloatTensor(outs)
        outs = torch.reshape(outs, (-1, constant.pose_size + constant.au_size))
        outs = self.dset.rescale_y(outs)
        return outs