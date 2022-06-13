import constants.constants as constants
from models.model5_Conditional_GAN.generating import GenerateModel5
from models.model5_Conditional_GAN.model import Generator as model5
from models.model5_Conditional_GAN.training import TrainModel5

def init_model_5(task):
    if(task == "train"):
        train = TrainModel5(gan=True)
        constants.train_model = train.train_model
    elif(task == "generate"):
        constants.model = model5
        generator = GenerateModel5()
        constants.generate_motion = generator.generate_motion