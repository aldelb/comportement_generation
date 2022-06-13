import constants.constants as constants
from models.speech_to_head.model import AutoEncoder as model
from models.speech_to_head.generating import GenerateModel10
from models.speech_to_head.training import TrainModel10

def init_model_10(task):
    if(task == "train"):
        train = TrainModel10(gan=False)
        constants.train_model = train.train_model
    elif(task == "generate"):
        constants.model = model
        generator = GenerateModel10()
        constants.generate_motion = generator.generate_motion