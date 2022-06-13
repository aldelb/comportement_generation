import constants.constants as constants
from models.model2_skip_connectivity.training import TrainModel2
from models.model2_skip_connectivity.generating import GenerateModel2
from models.model2_skip_connectivity.model import AutoEncoder as model2

def init_model_2(task):
    if(task == "train"):
        train = TrainModel2(gan=False)
        constants.train_model = train.train_model
    else:
        constants.model = model2
        generator = GenerateModel2()
        constants.generate_motion = generator.generate_motion