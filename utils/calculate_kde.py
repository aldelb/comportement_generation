import argparse
from genericpath import isdir
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import constant
from torch_dataset import TestSet
from utils.model_utils import find_model

from utils.params_utils import read_params

def calculate_kde(test_set, path_data_out, path_evaluation):

    gened_seqs = []
    for file in os.listdir(path_data_out):
        pd_file = pd.read_csv(path_data_out + file)
        pd_file = pd_file[["gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry", "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r",
                         "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]]
        gened_seqs.append(pd_file)
    
    gened_frames = np.concatenate(gened_seqs, axis=0)
    gened_frames[5:]
    real_frames = np.concatenate(test_set.Y_final_ori, axis=0)

    means = []
    ses = []

    #for i in range(1, 4):
    i = 1
    print("="*10, "step ", i, "="*10)
    params = {'bandwidth': np.logspace(-1, 1, 10)} #[ 0.1,  0.16681005,  0.27825594,  0.46415888,  0.77426368, 1.29154967,  2.15443469,  3.59381366,  5.9948425 , 10.        ])
    print("Grid search for bandwith parameter of Kernel Density...")
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=3)
    grid.fit(gened_frames)

    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

    scores = grid.best_estimator_.score_samples(real_frames)

    means.append(np.mean(scores))
    ses.append(np.std(scores)/np.sqrt(len(test_set)))
    print("mean ", means)
    print("ses ", ses)

    df = pd.DataFrame([*zip(means, ses)], columns=['mean', 'se'])
    df.to_csv(path_evaluation + "eval"+i+".csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-epoch', help='number of epoch of recovred model', default=10)
    args = parser.parse_args()
    read_params(args.params)

    model_file = find_model(int(args.epoch)) 

    path_data_out = constant.dir_path + constant.output_path + model_file[0:-3] + "/"
    if(not isdir(path_data_out)):
        raise Exception(path_data_out + "is not a directory")

    path_evaluation = constant.dir_path + constant.evaluation_path + model_file[0:-3] + "/"
    if(not isdir(path_evaluation)):
        os.makedirs(path_evaluation, exist_ok=True)

    test_set = TestSet()

    calculate_kde(test_set, path_data_out, path_evaluation)
