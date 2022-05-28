import torch
from models.GenerateClass import Generate

class GenerateModel9(Generate):
    def __init__(self):
        super(GenerateModel9, self).__init__()

    def generate_motion(self, model, inputs):
            inputs = self.reshape_input_pose(inputs)
            input_eye, input_pose_r, input_au = self.separate_openface_features(inputs, dim=1)
            with torch.no_grad():
                output_pose_r = model.forward(input_pose_r)
            outs = self.reshape_output(input_eye, output_pose_r, input_au)
            return outs
