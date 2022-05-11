import torch
import constant
from models.model1_simple_autoencoder.model import AutoEncoder as model1
from models.model2_skip_connectivity.model import AutoEncoder as model2
from models.model3_two_decoders.model import AutoEncoder as model3
from models.model4_GAN_autoencoders.model import Generator as model4
from models.model5_Conditional_GAN.model import Generator as model5
from models.speech_to_speech.model import AutoEncoder as model6
from models.pose_to_pose.model import AutoEncoder as model7

def find_model(epoch):
    model_file = constant.model_path + "epoch"
    model_file += f"_{epoch}.pt"
    return model_file

def load_model(param_path):
    # Load parameters
    if(constant.model_number == 1):
        model = model1()
    elif(constant.model_number == 2):
        model = model2()
    elif(constant.model_number == 3):
        model = model3()
    elif(constant.model_number == 4):
        model = model4()
    elif(constant.model_number == 5):
        model = model5()
    elif(constant.model_number == 6):
        model = model6()
    elif(constant.model_number == 7):
        model = model7()
    else:
        raise Exception("Model ", constant.model_number, " does not exist")

    model.load_state_dict(torch.load(param_path))
    return model

def saveModel(model, epoch, saved_path):
    torch.save(model.state_dict(), f'{saved_path}epoch_{epoch}.pt')
