import constants.constants as constants
from models.speech_to_head_GAN.model import AutoEncoder as model
from models.speech_to_head_GAN.generating import GenerateModel11
from models.speech_to_head_GAN.training import TrainModel11

def init_model_11(task):
    if(task == "train"):
        train = TrainModel11(gan=False)
        constants.train_model = train.train_model
    elif(task == "generate"):
        constants.model = model
        generator = GenerateModel11()
        constants.generate_motion = generator.generate_motion