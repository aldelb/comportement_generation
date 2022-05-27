import torch
import constants.constants as constants
from torch_dataset import TrainSet


def generate_motion_2(model, inputs, noise=None):

    dset = TrainSet()
    # Config input
    inputs = dset.scale_x(inputs)
    inputs = torch.FloatTensor(inputs).unsqueeze(1)
    inputs = torch.reshape(inputs, (-1, inputs.shape[2], inputs.shape[0]))
    with torch.no_grad():
        outs = model.forward(inputs).squeeze(1).numpy()
    outs = torch.FloatTensor(outs).unsqueeze(1)
    outs = torch.reshape(outs, (-1, constants.pose_size + constants.au_size))
    outs = dset.rescale_y(outs)
    return outs
