from matplotlib import pyplot as plt
import torch
import os
from datetime import date
import configparser
import constant
from models.model1_simple_autoencoder.model import write_model_1
from models.model2_skip_connectivity.model import write_model_2
from models.model3_two_decoders.model import write_model_3
from models.model4_GAN_autoencoders.model import write_model_4
from models.model5_Conditional_GAN.model import write_model_5
from models.pose_to_pose.model import write_model_7
from models.speech_to_speech.model import write_model_6

config = configparser.RawConfigParser()

def read_params(file):

    config.read(file)

    # --- model type params
    constant.model_number =  config.getint('MODEL_TYPE','model')
    try :
        constant.unroll_steps =  config.getboolean('MODEL_TYPE','unroll_steps')
    except:
        constant.unroll_steps =  config.getint('MODEL_TYPE','unroll_steps')
    constant.layer =  config.get('MODEL_TYPE','layer')
    constant.hidden_size =  config.getint('MODEL_TYPE','hidden_size') 
    constant.kernel_size = config.getint('MODEL_TYPE','kernel_size') 
    constant.first_kernel_size = config.getint('MODEL_TYPE','first_kernel_size') 
    constant.padding_size = int((constant.kernel_size - 1)/2)
    constant.first_padding_size = int((constant.first_kernel_size - 1)/2)
    constant.dropout =  config.getfloat('MODEL_TYPE','dropout') 

    datasets = config.get('PATH','datasets')
    constant.datasets = datasets.split(",")
    constant.dir_path = config.get('PATH','dir_path')
    constant.data_path = config.get('PATH','data_path')
    constant.saved_path = config.get('PATH','saved_path')
    constant.output_path = config.get('PATH','output_path')
    constant.evaluation_path = config.get('PATH','evaluation_path')
    constant.model_path = config.get('PATH','model_path')

    # --- Training params
    constant.n_epochs =  config.getint('TRAIN','n_epochs')
    constant.batch_size = config.getint('TRAIN','batch_size')
    constant.d_lr =  config.getfloat('TRAIN','d_lr')
    constant.g_lr =  config.getfloat('TRAIN','g_lr')
    constant.log_interval =  config.getint('TRAIN','log_interval')

    # --- Data params
    constant.noise_size = config.getint('DATA','noise_size')
    constant.pose_size = config.getint('DATA','pose_size') 
    constant.eye_size = config.getint('DATA','eye_size')
    constant.pose_t_size = config.getint('DATA','pose_t_size')
    constant.pose_r_size = config.getint('DATA','pose_r_size')
    constant.au_size = config.getint('DATA','au_size') 
    constant.derivative = config.getboolean('DATA','derivative')

    constant.opensmile_columns = get_config_columns('opensmile_columns')
    constant.selected_opensmile_columns = get_config_columns('selected_opensmile_columns')
    constant.openface_columns = get_config_columns('openface_columns')

    base_size = len(constant.selected_opensmile_columns) 

    if constant.derivative:
        constant.prosody_size = base_size * 3
    else:
        constant.prosody_size = base_size

    constant.selected_os_index_columns = []
    for column in constant.selected_opensmile_columns:
        constant.selected_os_index_columns.append(constant.opensmile_columns.index(column))


def get_config_columns(list):
    list_items = config.items(list)
    columns = []
    for _, column_name in list_items:
        columns.append(column_name)
    return columns

def write_params(f, title, params):
    f.write(f"# --- {title}\n")
    for argument in params.keys() :
        f.write(f"{argument} : {params[argument]}\n\n")

def save_params(saved_path, model, D = None):
    path_params = {"saved path" : saved_path}
    training_params = {
        "n_epochs" : constant.n_epochs,
        "batch_size" : constant.batch_size,
        "d_lr" : constant.d_lr,
        "g_lr" : constant.g_lr}

    model_params = {
        "model" : constant.model_number,
        "unroll_steps" : constant.unroll_steps,
        "layer" : constant.layer,
        "hidden_size" : constant.hidden_size,
        "first_kernel_size" : constant.first_kernel_size,
        "kernel_size" : constant.kernel_size,
        "first_padding_size" : constant.first_padding_size,
        "padding_size" : constant.padding_size,
        "dropout" : constant.dropout}

    data_params = {
        "log_interval" : constant.log_interval,
        "noise_size" : constant.noise_size,
        "prosody_size" : constant.prosody_size,
        "pose_size" : constant.pose_size,
        "au_size" : constant.au_size,
        "column keep in opensmile" : constant.selected_opensmile_columns,
        "derivative" : constant.derivative}

    file_path = saved_path + "parameters.txt"
    f = open(file_path, "w")
    write_params(f, "Model params", model_params)
    write_params(f, "Path params", path_params)
    write_params(f, "Training params", training_params)
    write_params(f, "Data params", data_params)

    f.write("-"*10 + "Models" + "-"*10 + "\n")
    f.close()

    if(constant.model_number == 1):
        write_model_1(file_path, model, D)
    elif(constant.model_number == 2):
        write_model_2(file_path, model, D)
    elif(constant.model_number == 3):
        write_model_3(file_path, model, D)
    elif(constant.model_number == 4):
        write_model_4(file_path, model, D)
    elif(constant.model_number == 5):
        write_model_5(file_path, model, D)
    elif(constant.model_number == 6):
        write_model_6(file_path, model, D)
    elif(constant.model_number == 7):
        write_model_7(file_path, model, D)
    else:
        raise Exception("Model ", constant.model_number, " does not exist")
    

def create_saved_path(config_file, id):
    # * Create dir for store models, hist and params
    today = date.today().strftime("%d-%m-%Y")
    saved_path = constant.dir_path + constant.saved_path 
    str_dataset = ""
    for dataset in constant.datasets:
        str_dataset += dataset + "_"

    dir_path = f"{today}_{str_dataset}"
    if(id == "0"):
        i = 1
        while(os.path.isdir(saved_path + dir_path + f"{i}")):
            i = i+1
        dir_path += f"{i}/"
    else:
        dir_path += f"{id}/"
    saved_path += dir_path
    os.makedirs(saved_path, exist_ok=True)
    config.set('PATH','model_path', dir_path)
    with open(config_file, 'w') as configfile:
        config.write(configfile)
    return saved_path