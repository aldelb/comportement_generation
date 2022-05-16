import argparse
from genericpath import isdir
import os
import numpy as np
import constant
from utils.create_final_file import createFinalFile
from models.model1_simple_autoencoder.generating import generate_motion_1
from models.model2_skip_connectivity.generating import generate_motion_2
from models.model3_two_decoders.generating import generate_motion_3
from models.model4_GAN_autoencoders.generating import generate_motion_4
from models.model5_Conditional_GAN.generating import generate_motion_5
from utils.model_utils import find_model, load_model
from utils.params_utils import read_params
from torch_dataset import TestSet
import pandas as pd

def generate_motion(model, input):
    if(constant.model_number == 1):
        return generate_motion_1(model, input)
    elif(constant.model_number == 2):
        return generate_motion_2(model, input)
    elif(constant.model_number == 3):
        return generate_motion_3(model, input)
    elif(constant.model_number == 4):
        return generate_motion_4(model, input)
    elif(constant.model_number == 5):
        return generate_motion_5(model, input)
    else:
        raise Exception("Model ", constant.model_number, " does not exist")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-epoch', help='number of epoch of recovred model', default=40)
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
        input, _ = data
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

            #file_out = path_data_out + key + "-" + str(current_part) + ".csv"
            timestamp = np.array(test_set.getInterval(index)[1][:,0])
            out = np.concatenate((timestamp.reshape(-1,1), out), axis=1)
            df = pd.DataFrame(data = out, columns = columns)
            #df.set_index('timestamp', inplace = True)
            df_list.append(df)
            #df.to_csv(file_out, sep=',')
            current_part += 1