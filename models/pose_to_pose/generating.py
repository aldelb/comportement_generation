import torch
from models.GenerateClass import Generate

class GenerateModel7(Generate):
    def __init__(self):
        super(GenerateModel7, self).__init__()

    def generate_motion(self, model, pose):
        pose = self.reshape_pose(pose)
        with torch.no_grad():
            outs = model.forward(pose)
        outs = self.reshape_single_output(outs)
        return outs
