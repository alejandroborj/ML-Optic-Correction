from model_training import train_model

set_name = 'data'
TRAIN_MODEL = True

if TRAIN_MODEL == True:
    train_model(set_name, MERGE=True)