import constants.constants as constants
from models.pose_to_pose_multiple_decoders.model import AutoEncoder as model
from models.pose_to_pose_multiple_decoders.generating import GenerateModel8
from models.pose_to_pose_multiple_decoders.training import TrainModel8

def init_model_8(task):
    if(task == "train"):
        train = TrainModel8(gan=False)
        constants.train_model = train.train_model
    elif(task == "generate"):
        constants.model = model
        generator = GenerateModel8()
        constants.generate_motion = generator.generate_motion