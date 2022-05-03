import torch
import constant
from torch_dataset import TrainSet

def generate_motion_4(model, inputs):

    dset = TrainSet()
    # Config input
    inputs = dset.scale_x(inputs)
    inputs = torch.FloatTensor(inputs).unsqueeze(1)
    inputs = torch.reshape(inputs, (-1, inputs.shape[2], inputs.shape[0]))
    with torch.no_grad():
        output_pose, output_au = model.forward(inputs)
        output_pose = output_pose.squeeze(1).numpy()
        output_au = output_au.squeeze(1).numpy()
    
    output_pose = torch.FloatTensor(output_pose).unsqueeze(1)
    output_pose = torch.reshape(output_pose, (-1, constant.pose_size))

    output_au = torch.FloatTensor(output_au).unsqueeze(1)
    output_au = torch.reshape(output_au, (-1, constant.au_size))
    
    #concatener les deux dans outs
    outs = torch.cat((output_pose, output_au), 1)
    outs = dset.rescale_y(outs)
    return outs

