import dlib, sys, os
import pandas as pd
import streamlit as st
from stqdm import stqdm
from sklearn.metrics.pairwise import cosine_similarity

import utils.helper_functions as helper_functions


XAI_APPROACHES=[
    'Minus Single',
    'Minus Greedy',
    'Plus Single',
    'Plus Greedy',
    'Average'
]
MODELS = [
    # 'VGG-Face', 
    # 'OpenFace', 
    'ArcFace'
]
METRICS=["cosine"]
BACKENDS=["mtcnn"]

THRESHOLDS={
        # 'VGG-Face': 0.45, 
        # 'OpenFace': 0.35, 
        'ArcFace' : 0.31,  
}

# For PIC-Score
GENUINE_ARCFACE_SIM_SCORES_PATH='./artifacts/embeddings_genuine.npy'
IMPOSTER_ARCFACE_SIM_SCORES_PATH='./artifacts/embeddings_impostor.npy'



#################### <handle tmp folder> ####################
APP_TMP="./output/tmp"
helper_functions.recreate_folder(APP_TMP)
#################### </handle tmp folder> ###################

#################### <create uploaded imgs folder> ####################
APP_UPLOADED="./output/uploaded_images"
try:    os.makedirs(APP_UPLOADED)
except: pass
#################### </create uploaded imgs folder> ####################

#################### <create output folder> ####################
APP_OUTPUT="./output/output_images"
try:    os.makedirs(APP_OUTPUT)
except: pass
#################### </create output folder> ####################

#################### <create face imgs folder> ####################
APP_FACE_OUTPUT="./output/face_images"
try:    os.makedirs(APP_FACE_OUTPUT)
except: pass
#################### </create face imgs folder> ####################

#################### <create models folder> ####################
APP_MODELS="./models"
try:    os.makedirs(APP_MODELS)
except: pass
#################### </create models folder> ####################

################### <landmarks assets> ######################
DLIB_LM_MODEL_PATH='./models/shape_predictor_68_face_landmarks.dat'
DLIB_LM_MODEL = dlib.shape_predictor(DLIB_LM_MODEL_PATH)

# wget http://arunponnusamy.com/files/mmod_human_face_detector.dat
DLIB_FACE_DETECT_MODEL_PATH="./models/mmod_human_face_detector.dat"
DLIB_FACE_DETECT_MODEL = dlib.cnn_face_detection_model_v1(DLIB_FACE_DETECT_MODEL_PATH)
################### </landmarks assets> #####################

