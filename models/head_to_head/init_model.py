import constants.constants as constants
from models.head_to_head.model import AutoEncoder as model
from models.head_to_head.generating import GenerateModel9
from models.head_to_head.training import TrainModel9

def init_model_9(task):
    if(task == "train"):
        train = TrainModel9(gan=False)
        constants.train_model = train.train_model
    elif(task == "generate"):
        constants.model = model
        generator = GenerateModel9()
        constants.generate_motion = generator.generate_motion