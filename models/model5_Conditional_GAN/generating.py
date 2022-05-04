import torch
import constant
from torch_dataset import TrainSet
from utils.noise_generator import NoiseGenerator

def generate_motion_5(model, inputs):
    dset = TrainSet()
    inputs = dset.scale_x(inputs)
    inputs = torch.FloatTensor(inputs).unsqueeze(1)
    inputs = torch.reshape(inputs, (-1, inputs.shape[2], inputs.shape[0]))
    
    noise_g = NoiseGenerator()
    noise = noise_g.gaussian_variating(T=inputs.shape[0], F=40, size=model.n_size, allow_indentical=True)
    noise = torch.FloatTensor(noise).unsqueeze(1)

    with torch.no_grad():
        outs = model.forward(inputs, noise).squeeze(1).numpy()
    outs = torch.FloatTensor(outs).unsqueeze(1)
    outs = torch.reshape(outs, (-1, constant.pose_size + constant.au_size))
    outs = dset.rescale_y(outs)
    return outs

