import torch
import constants.constants as constants
from models.GenerateClass import Generate
from torch_dataset import TrainSet



class GenerateModel11(Generate):
    def __init__(self):
        super(GenerateModel11, self).__init__()

    def generate_motion(self, model, inputs, targets):
        inputs = self.reshape_input(inputs)
        targets = self.reshape_input_pose(targets)
        input_eye, input_pose_r, input_au = self.separate_openface_features(targets, dim=1)
        with torch.no_grad():
            output_pose_r = model.forward(inputs)
        outs = self.reshape_output(input_eye, output_pose_r, input_au)
        return outs