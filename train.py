import argparse
import constant
from models.model1_simple_autoencoder.training import train_model_1
from models.model2_skip_connectivity.training import TrainModel2
from models.model3_multiple_decoders.training import TrainModel3

from models.model4_GAN_autoencoders.training import TrainModel4
from models.model5_Conditional_GAN.training import train_model_5
from models.pose_to_pose_multiple_decoders.training import TrainModel8
from models.speech_to_speech.training import train_model_speech_to_speech
from models.pose_to_pose.training import TrainModel7
from utils.params_utils import create_saved_path, read_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-id', help='Path to save result and models', default="0")
    args = parser.parse_args()
    read_params(args.params)

    constant.saved_path = create_saved_path(args.params, args.id)

    if(constant.model_number == 1):
        train_model_1()
    elif(constant.model_number == 2):
        train = TrainModel2(gan=False)
        train.train_model()
    elif(constant.model_number == 3):
        train = TrainModel3(gan=False)
        train.train_model()
    elif(constant.model_number == 4):
        train = TrainModel4(gan=True)
        train.train_model()
    elif(constant.model_number == 5):
        train_model_5()
    elif(constant.model_number == 6):
        train_model_speech_to_speech()
    elif(constant.model_number == 7):
        train = TrainModel7(gan=False)
        train.train_model()
    elif(constant.model_number == 8):
        train = TrainModel8(gan=False)
        train.train_model()
    else:
        raise Exception("Model ", constant.model_number, " does not exist")

    