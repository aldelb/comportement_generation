import torch
import constants.constants as constants
from models.GenerateClass import Generate
from torch_dataset import TrainSet



class GenerateModel10(Generate):
    def __init__(self):
        super(GenerateModel10, self).__init__()

    def generate_motion(self, model, inputs, target):
        inputs = self.reshape_input(inputs[0])
        input_eye, input_pose_t, input_pose_r, input_au = self.separate_openface_features(inputs[1], dim=1)
        with torch.no_grad():
            output_pose_r = model.forward(inputs)
        outs = self.reshape_output(input_eye, input_pose_t, output_pose_r, input_au)
        return outs