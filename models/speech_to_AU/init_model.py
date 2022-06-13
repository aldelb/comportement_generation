import constants.constants as constants
from models.speech_to_AU.training import TrainModel12
from models.speech_to_AU.generating import GenerateModel12
from models.speech_to_AU.model import AutoEncoder as model

def init_model_12(task):
    if(task == "train"):
        train = TrainModel12(gan=False)
        constants.train_model = train.train_model
    else:
        constants.model = model
        generator = GenerateModel12()
        constants.generate_motion = generator.generate_motion