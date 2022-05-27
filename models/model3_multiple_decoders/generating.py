import torch
import constants.constants as constants
from models.GenerateClass import Generate
from torch_dataset import TrainSet



class GenerateModel3(Generate):
    def __init__(self):
        super(GenerateModel3, self).__init__()

    def generate_motion(self, model, inputs):
        inputs = self.reshape_input(inputs)
        with torch.no_grad():
            output_eye, output_pose_t, output_pose_r, output_au = model.forward(inputs)
        outs = self.reshape_output(output_eye, output_pose_t, output_pose_r, output_au)
        return outs