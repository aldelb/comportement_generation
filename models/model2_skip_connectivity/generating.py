import torch
import constant
from models.GenerateClass import Generate
from torch_dataset import TrainSet


class GenerateModel2(Generate):
    def __init__(self):
        super(GenerateModel2, self).__init__()

    def generate_motion(self, model, inputs):
        inputs = self.reshape_input(inputs)
        with torch.no_grad():
            outs = model.forward(inputs)
        outs = self.reshape_single_output(outs)
        return outs
