import constants.constants as constants
from models.speech_to_speech.training import train_model_speech_to_speech

def init_model_6(task):
    if(task == "train"):
        constants.train_model = train_model_speech_to_speech
    elif(task == "generate"):
        raise Exception("No generation possible with model 6")