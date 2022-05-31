import torch
from models.GenerateClass import Generate

class GenerateModel2(Generate):
    def __init__(self):
        super(GenerateModel2, self).__init__()

    def generate_motion(self, model, prosody):
        prosody = self.reshape_prosody(prosody)
        with torch.no_grad():
            outs = model.forward(prosody)
        outs = self.reshape_single_output(outs)
        return outs
