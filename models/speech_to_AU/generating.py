import torch
from models.GenerateClass import Generate

class GenerateModel12(Generate):
    def __init__(self):
        super(GenerateModel12, self).__init__()

    def generate_motion(self, model, prosody, pose):
        prosody = self.reshape_prosody(prosody)
        pose = self.reshape_pose(pose)
        input_eye, input_pose_r, input_au = self.separate_openface_features(pose)
        with torch.no_grad():
            output_pose_au = model.forward(prosody)
        outs = self.reshape_output(input_eye, input_pose_r, output_pose_au)
        return outs