import constants.constants as constants
from models.model4_GAN_autoencoders.generating import GenerateModel4
from models.model4_GAN_autoencoders.model import Generator as model4
from models.model4_GAN_autoencoders.training import TrainModel4


def init_model_4(task):
    if(task == "train"):
        train = TrainModel4(gan=True)
        constants.train_model = train.train_model
    elif(task == "generate"):
        constants.model = model4
        generator = GenerateModel4()
        constants.generate_motion = generator.generate_motion