import argparse
from genericpath import isdir
import os
import numpy as np
import constant
from models.model3_multiple_decoders.generating import GenerateModel3
from models.pose_to_pose.generating import GenerateModel7
from models.pose_to_pose_multiple_decoders.generating import GenerateModel8
from utils.create_final_file import createFinalFile
from models.model1_simple_autoencoder.generating import generate_motion_1
from models.model2_skip_connectivity.generating import GenerateModel2
from models.model4_GAN_autoencoders.generating import GenerateModel4
from models.model5_Conditional_GAN.generating import generate_motion_5
from utils.model_utils import find_model, load_model
from utils.params_utils import read_params
from torch_dataset import TestSet
import pandas as pd

def generate_motion(model, input):
    if(constant.model_number == 1):
        return generate_motion_1(model, input)
    elif(constant.model_number == 2):
        generator = GenerateModel2()
        return generator.generate_motion(model, input)
    elif(constant.model_number == 3):
        generator = GenerateModel3()
        return generator.generate_motion(model, input)
    elif(constant.model_number == 4):
        generator = GenerateModel4()
        return generator.generate_motion(model, input)
    elif(constant.model_number == 5):
        return generate_motion_5(model, input)
    elif(constant.model_number == 7):
        generator = GenerateModel7()
        return generator.generate_motion(model, input)
    elif(constant.model_number == 8):
        generator = GenerateModel8()
        return generator.generate_motion(model, input)
    else:
        raise Exception("Model ", constant.model_number, " does not exist")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-epoch', help='number of epoch of recovred model', default=9)
    parser.add_argument('-video', help='wich video we want to generate', default="all")
    args = parser.parse_args()
    read_params(args.params)

    model_file = find_model(int(args.epoch)) 
    model = load_model(constant.dir_path + constant.saved_path + model_file)

    path_data_out = constant.dir_path + constant.output_path + model_file[0:-3] + "/"
    if(not isdir(path_data_out)):
        os.makedirs(path_data_out, exist_ok=True)

    test_set = TestSet()

    gened_seqs = []
    columns = constant.openface_columns

    current_part = 0 
    current_key = ""
    df_list = []

    for index, data in enumerate(test_set):
        input, target = data
        if(constant.model_number == 7 or constant.model == 8):
            input = target
        key = test_set.getInterval(index)[0]
        if(args.video == "all" or key == args.video + "[0]"):
            if(current_key != key): #process of a new video
                if(current_key != ""):
                    createFinalFile(path_data_out, current_key, df_list)
                print("Generation of video", key[0:-3] , "...")
                current_part = 0
                current_key = key
                df_list = []

            if(constant.prosody_size == 1):
                input = np.reshape(input, input.shape + (1,))

            out = generate_motion(model, input)
            timestamp = np.array(test_set.getInterval(index)[1][:,0])
            out = np.concatenate((timestamp.reshape(-1,1), out), axis=1)
            df = pd.DataFrame(data = out, columns = columns)
            df_list.append(df)
            current_part += 1

    createFinalFile(path_data_out, current_key, df_list)