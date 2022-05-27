import configparser
import constants.constants as constants
import os
from datetime import date

from models.head_to_head.generating import GenerateModel9
from models.head_to_head.model import write_model_9
from models.head_to_head.training import TrainModel9
from models.model1_simple_autoencoder.generating import generate_motion_1
from models.model1_simple_autoencoder.model import write_model_1
from models.model1_simple_autoencoder.training import train_model_1
from models.model2_skip_connectivity.generating import GenerateModel2
from models.model2_skip_connectivity.model import write_model_2
from models.model2_skip_connectivity.training import TrainModel2
from models.model3_multiple_decoders.generating import GenerateModel3
from models.model3_multiple_decoders.model import write_model_3
from models.model3_multiple_decoders.training import TrainModel3
from models.model4_GAN_autoencoders.generating import GenerateModel4
from models.model4_GAN_autoencoders.model import write_model_4
from models.model4_GAN_autoencoders.training import TrainModel4
from models.model5_Conditional_GAN.generating import generate_motion_5
from models.model5_Conditional_GAN.model import write_model_5
from models.model5_Conditional_GAN.training import train_model_5
from models.pose_to_pose.generating import GenerateModel7
from models.pose_to_pose.model import write_model_7
from models.pose_to_pose.training import TrainModel7
from models.pose_to_pose_multiple_decoders.generating import GenerateModel8
from models.pose_to_pose_multiple_decoders.model import write_model_8
from models.pose_to_pose_multiple_decoders.training import TrainModel8
from models.speech_to_head.generating import GenerateModel10
from models.speech_to_head.model import write_model_10
from models.speech_to_head.training import TrainModel10
from models.speech_to_speech.model import write_model_6
from models.speech_to_speech.training import train_model_speech_to_speech
from models.model1_simple_autoencoder.model import AutoEncoder as model1
from models.model2_skip_connectivity.model import AutoEncoder as model2
from models.model3_multiple_decoders.model import AutoEncoder as model3
from models.model4_GAN_autoencoders.model import Generator as model4
from models.model5_Conditional_GAN.model import Generator as model5
from models.speech_to_speech.model import AutoEncoder as model6
from models.pose_to_pose.model import AutoEncoder as model7
from models.pose_to_pose_multiple_decoders.model import AutoEncoder as model8
from models.head_to_head.model import AutoEncoder as model9
from models.speech_to_head.model import AutoEncoder as model10


config = configparser.RawConfigParser()

def read_params(file, task="train", id=None):

    config.read(file)

    # --- model type params
    constants.model_number =  config.getint('MODEL_TYPE','model')
    try :
        constants.unroll_steps =  config.getboolean('MODEL_TYPE','unroll_steps')
    except:
        constants.unroll_steps =  config.getint('MODEL_TYPE','unroll_steps')
    constants.layer =  config.get('MODEL_TYPE','layer')
    constants.hidden_size =  config.getint('MODEL_TYPE','hidden_size') 
    constants.kernel_size = config.getint('MODEL_TYPE','kernel_size') 
    constants.first_kernel_size = config.getint('MODEL_TYPE','first_kernel_size') 
    constants.padding_size = int((constants.kernel_size - 1)/2)
    constants.first_padding_size = int((constants.first_kernel_size - 1)/2)
    constants.dropout =  config.getfloat('MODEL_TYPE','dropout') 

    datasets = config.get('PATH','datasets')
    constants.datasets = datasets.split(",")
    constants.dir_path = config.get('PATH','dir_path')
    constants.data_path = config.get('PATH','data_path')
    constants.saved_path = config.get('PATH','saved_path')
    constants.output_path = config.get('PATH','output_path')
    constants.evaluation_path = config.get('PATH','evaluation_path')
    constants.model_path = config.get('PATH','model_path')

    # --- Training params
    constants.n_epochs =  config.getint('TRAIN','n_epochs')
    constants.batch_size = config.getint('TRAIN','batch_size')
    constants.d_lr =  config.getfloat('TRAIN','d_lr')
    constants.g_lr =  config.getfloat('TRAIN','g_lr')
    constants.log_interval =  config.getint('TRAIN','log_interval')

    # --- Data params
    constants.noise_size = config.getint('DATA','noise_size')
    constants.pose_size = config.getint('DATA','pose_size') 
    constants.eye_size = config.getint('DATA','eye_size')
    constants.pose_t_size = config.getint('DATA','pose_t_size')
    constants.pose_r_size = config.getint('DATA','pose_r_size')
    constants.au_size = config.getint('DATA','au_size') 
    constants.derivative = config.getboolean('DATA','derivative')

    constants.opensmile_columns = get_config_columns('opensmile_columns')
    constants.selected_opensmile_columns = get_config_columns('selected_opensmile_columns')
    constants.openface_columns = get_config_columns('openface_columns')

    base_size = len(constants.selected_opensmile_columns) 

    if constants.derivative:
        constants.prosody_size = base_size * 3
    else:
        constants.prosody_size = base_size

    constants.selected_os_index_columns = []
    for column in constants.selected_opensmile_columns:
        constants.selected_os_index_columns.append(constants.opensmile_columns.index(column))

    if(task == "train"):
        constants.saved_path = create_saved_path(file, id)
    set_model_function(constants.model_number, task)

def set_model_function(model_number, task):
    if(model_number == 1): #simple auto encoder
        if(task == "train"):
            constants.write_model = write_model_1
            constants.train_model = train_model_1()
        elif(task == "generate"):
            constants.model = model1
            constants.generate_motion = generate_motion_1

    elif(model_number == 2): #simple autoencoder with skip connectivity
        if(task == "train"):
            constants.write_model = write_model_2
            train = TrainModel2(gan=False)
            constants.train_model = train.train_model
        elif(task == "generate"):
            constants.model = model2
            generator = GenerateModel2()
            constants.generate_motion = generator.generate_motion

    elif(model_number == 3): #multiple decoders
        if(task == "train"):
            constants.write_model = write_model_3
            train = TrainModel3(gan=False)
            constants.train_model = train.train_model
        elif(task == "generate"):
            constants.model = model3
            generator = GenerateModel3()
            constants.generate_motion = generator.generate_motion

    elif(model_number == 4): #autoencoders GAN
        if(task == "train"):
            constants.write_model = write_model_4
            train = TrainModel4(gan=True)
            constants.train_model = train.train_model
        elif(task == "generate"):
            constants.model = model4
            generator = GenerateModel4()
            constants.generate_motion = generator.generate_motion

    elif(model_number == 5): #GAN
        if(task == "train"):
            constants.write_model = write_model_5
            constants.train_model = train_model_5
        elif(task == "generate"):
            constants.model = model5
            constants.generate_motion = generate_motion_5

    elif(model_number == 6): #speech to speech
        if(task == "train"):
            constants.write_model = write_model_6
            constants.train_model = train_model_speech_to_speech
        elif(task == "generate"):
            raise Exception("No generation possible with model 6")

    elif(model_number == 7): #pose to pose
        if(task == "train"):
            constants.write_model = write_model_7
            train = TrainModel7(gan=False)
            constants.train_model = train.train_model
        elif(task == "generate"):
            constants.model = model7
            generator = GenerateModel7()
            constants.generate_motion = generator.generate_motion

    elif(model_number == 8): #pose to pose multiple decoders
        if(task == "train"):
            constants.write_model = write_model_8
            train = TrainModel8(gan=False)
            constants.train_model = train.train_model
        elif(task == "generate"):
            constants.model = model8
            generator = GenerateModel8()
            constants.generate_motion = generator.generate_motion

    elif(model_number == 9): #head to head
        if(task == "train"):
            constants.write_model = write_model_9
            train = TrainModel9(gan=False)
            constants.train_model = train.train_model
        elif(task == "generate"):
            constants.model = model9
            generator = GenerateModel9()
            constants.generate_motion = generator.generate_motion
            
    elif(model_number == 10): #speech to head
        if(task == "train"):
            constants.write_model = write_model_10
            train = TrainModel10(gan=False)
            constants.train_model = train.train_model
        elif(task == "generate"):
            constants.model = model10
            generator = GenerateModel10()
            constants.generate_motion = generator.generate_motion
    else:
        raise Exception("Model ", model_number, " does not exist")

def get_config_columns(list):
    list_items = config.items(list)
    columns = []
    for _, column_name in list_items:
        columns.append(column_name)
    return columns

def create_saved_path(config_file, id):
    # * Create dir for store models, hist and params
    today = date.today().strftime("%d-%m-%Y")
    saved_path = constants.dir_path + constants.saved_path 
    str_dataset = ""
    for dataset in constants.datasets:
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