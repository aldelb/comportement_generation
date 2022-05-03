import argparse
import constant
from models.model1_simple_autoencoder.training import train_model_1
from models.model2_skip_connectivity.training import train_model_2
from models.model3_two_decoders.training import train_model_3
from models.model4_GAN_autoencoders.training import train_model_4

from utils.params_utils import create_saved_path, read_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    args = parser.parse_args()
    read_params(args.params)

    constant.saved_path = create_saved_path(args.params)
    
    if(constant.model_number == 1):
        train_model_1()
    elif(constant.model_number == 2):
        train_model_2()
    elif(constant.model_number == 3):
        train_model_3()
    elif(constant.model_number == 4):
        train_model_4()
    else:
        raise Exception("Model ", constant.model_number, " does not exist")