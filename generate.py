import argparse
from genericpath import isdir
import os
import numpy as np
import constants.constants as constants
from constants.constants_utils import read_params
from utils.create_final_file import createFinalFile
from utils.model_utils import find_model, load_model
from torch_dataset import TestSet
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-epoch', help='number of epoch of recovred model', default=9)
    parser.add_argument('-video', help='wich video we want to generate', default="all")
    
    args = parser.parse_args()
    read_params(args.params, "generate")

    model_file = find_model(int(args.epoch)) 
    model = load_model(constants.dir_path + constants.saved_path + model_file)

    path_data_out = constants.dir_path + constants.output_path + model_file[0:-3] + "/"
    if(not isdir(path_data_out)):
        os.makedirs(path_data_out, exist_ok=True)

    test_set = TestSet()

    gened_seqs = []
    columns = constants.openface_columns

    current_part = 0 
    current_key = ""
    df_list = []

    for index, data in enumerate(test_set):
        input, target = data
        if(constants.model_number in [7,8,9]):
            input = target
        if(constants.model_number in [10]):
            input[0] = input
            input[1] = target
        key = test_set.getInterval(index)[0]
        if(args.video == "all" or key == args.video + "[0]"):
            if(current_key != key): #process of a new video
                if(current_key != ""):
                    createFinalFile(path_data_out, current_key, df_list)
                print("Generation of video", key[0:-3] , "...")
                current_part = 0
                current_key = key
                df_list = []

            if(constants.prosody_size == 1):
                input = np.reshape(input, input.shape + (1,))

            out = constants.generate_motion(model, input)
            timestamp = np.array(test_set.getInterval(index)[1][:,0])
            out = np.concatenate((timestamp.reshape(-1,1), out), axis=1)
            df = pd.DataFrame(data = out, columns = columns)
            df_list.append(df)
            current_part += 1

    createFinalFile(path_data_out, current_key, df_list)