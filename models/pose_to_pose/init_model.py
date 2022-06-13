import constants.constants as constants
from model import AutoEncoder as model
from models.pose_to_pose.generating import GenerateModel7
from models.pose_to_pose.training import TrainModel7

def init_model_7(task):
    if(task == "train"):
        train = TrainModel7(gan=False)
        constants.train_model = train.train_model
    elif(task == "generate"):
        constants.model = model
        generator = GenerateModel7()
        constants.generate_motion = generator.generate_motion