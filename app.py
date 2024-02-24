import pdb, os
import dlib
import numpy as np
import sys,os, cv2
import pandas as pd
from PIL import Image
from time import time
import streamlit as st
from stqdm import stqdm
from sklearn.metrics.pairwise import cosine_similarity

from utils import helper_functions
import utils.params  as params
import utils.app_utils as app_utils
import utils.blackbox_explaination.fvx_utils as fxv
from utils.optimal_matching_confidence import ConfidenceEstimator


MODELS   = params.MODELS
METRICS  = params.METRICS
# ATTRS_DF = params.ATTRS_DF
BACKENDS = params.BACKENDS
XAI_APPROACHES = params.XAI_APPROACHES


st.set_page_config(page_title="Image Comparison", page_icon=":camera:", layout="wide")

st.title("Image Comparison")
col1, col2 = st.columns(2)

# upload image-1
uploaded_file1 = col1.file_uploader("Upload Reference Image", type=["jpg", "jpeg", "png"])

# upload image-2
uploaded_file2 = col2.file_uploader("Upload Probe Image", type=["jpg", "jpeg", "png"])

if uploaded_file1 is not None:
    file_content = uploaded_file1.read()
    with open(f'{params.APP_UPLOADED}/{uploaded_file1.name}', 'wb') as f: f.write(file_content)

    image1 = Image.open(uploaded_file1).convert('RGB')

    image1_sz = image1.resize((300,300))# int(image1.height * 300 / image1.width)))
    col1.image(image1_sz, use_column_width=False)
    st.session_state.image1_sz = image1_sz
    st.session_state.uploaded_file1 = uploaded_file1
elif 'image1_sz' in st.session_state:
    col1.image(st.session_state.image1_sz, use_column_width=False)

if uploaded_file2 is not None:
    file_content = uploaded_file2.read()
    with open(f'{params.APP_UPLOADED}/{uploaded_file2.name}', 'wb') as f: f.write(file_content)

    image2 = Image.open(uploaded_file2).convert('RGB')

    image2_sz = image2.resize((300,300))# int(image1.height * 300 / image1.width)))
    col2.image(image2_sz, use_column_width=False )
    st.session_state.image2_sz = image2_sz
    st.session_state.uploaded_file2 = uploaded_file2
elif 'image2_sz' in st.session_state:
    col2.image(st.session_state.image2_sz, use_column_width=False)

if (('uploaded_file1' in st.session_state and st.session_state.uploaded_file1 is not None) and 
    ('uploaded_file2' in st.session_state and st.session_state.uploaded_file2 is not None)):
    st.markdown("***")

    img1_path=f'{params.APP_UPLOADED}/{st.session_state.uploaded_file1.name}'
    img2_path=f'{params.APP_UPLOADED}/{st.session_state.uploaded_file2.name}'


    left_co, cent_co,last_co = st.columns(3)
    with cent_co: 
        pair_label=st.selectbox(
                                label='', 
                                placeholder='Select Pair Label',
                                options=['genuine', 'imposter'],
                                key='pair_label'
                            )
        compare_button_clicked = st.button(
                                        label='Compare Photos', 
                                        key='compare'
                                    )
    if compare_button_clicked:



        rdf=app_utils.run_face_comparison(
                                f'{params.APP_UPLOADED}/{uploaded_file1.name}', 
                                f'{params.APP_UPLOADED}/{uploaded_file2.name}',
                                label=pair_label, models=MODELS, metrics=METRICS, backends=BACKENDS
        )
        rdf['decision_confidence'] = ConfidenceEstimator(
                                            params.GENUINE_ARCFACE_SIM_SCORES_PATH, 
                                            params.IMPOSTER_ARCFACE_SIM_SCORES_PATH
                                    ).confidence_single([rdf.similarity], 'gen' if pair_label=='genuine' else 'imp')
        st.session_state['binary_decision_df'] = rdf
        rdf.to_csv(f'{params.APP_TMP}/rdf.csv', index=False)
        st.dataframe(rdf)
        app_utils.display_comparison_result(rdf)
            
    else:
        try:
            rdf=pd.read_csv(f'{params.APP_TMP}/rdf.csv')
            st.dataframe(rdf)
            app_utils.display_comparison_result(rdf)
        except:
            pass

    selected_models = st.multiselect(
                            'Select Model',
                            options=MODELS,
                            placeholder='Select Face Recognition Model(s)',
                            key='models'
                        )


    selected_methods = st.multiselect(
                                'Select Method',
                                options=XAI_APPROACHES,  
                                placeholder='Select Explanation Method(s)',
                                key='methods'
                            )
    
    left_co, cent_co,last_co = st.columns(3)
    with cent_co: button_clicked = st.button('Generate Explanation')

    for single_model in MODELS:
        for single_method in XAI_APPROACHES:
            if f"{single_model}_{single_method}" in st.session_state and st.session_state[f"{single_model}_{single_method}"] is not None:
                st.subheader(f"Model: {single_model}")
                st.write(f"Method: {single_method}")
                st.image(st.session_state[f"{single_model}_{single_method}"], use_column_width=True)



    if 'display_df' in st.session_state and st.session_state['display_df'] is not None:
        st.subheader('All Results')
        st.dataframe(st.session_state['display_df'])

    if 'best_overall_result' in st.session_state and st.session_state['best_overall_result'] is not None:
        st.subheader('Most Important Features')
        st.dataframe(st.session_state['best_overall_result'])

    if 'most_imp_explanations' in st.session_state and st.session_state['most_imp_explanations'] is not None:
        # Print or use explanations as needed
        for explanation in st.session_state['most_imp_explanations']:
            st.write(f"- {explanation}")


    if 'least_overall_result' in st.session_state and st.session_state['least_overall_result'] is not None:
        st.subheader('Least Important Features')
        st.dataframe(st.session_state['least_overall_result'])

    if 'least_imp_explanations' in st.session_state and st.session_state['least_imp_explanations'] is not None:
        # Print or use explanations as needed
        for explanation in st.session_state['least_imp_explanations']:
            st.write(f"- {explanation}")

    if button_clicked:

        if len(selected_methods) == 0 or len(selected_models) == 0:
            st.error('Please select at least one model and one method')
            button_clicked = False
        else:

            hm_paths=[]
            final_imp_df=pd.DataFrame()
            
            for single_model in stqdm(selected_models):
                try:

                    # single_model, single_detector = option.split(' with ')
                    single_detector = 'mtcnn'
                    st.subheader(f"Model: {single_model}")


                    # setting the model_name as global variable in fvx_utils so we dont have to pass model_name each time as parameter
                    fxv.set_face_recognition_model(single_model, single_detector)

                    # since we have set the model_name as global variable, we dont have to pass it again
                    embeddings=app_utils.run_comparison_result(image1, image2, single_model)

                    img1_face_obj, img1_face_path=app_utils.crop_and_save_face(img1_path, single_model, single_detector, cropped_face_dir=f'{params.APP_FACE_OUTPUT}')
                    img2_face_obj, img2_face_path=app_utils.crop_and_save_face(img2_path, single_model, single_detector, cropped_face_dir=f'{params.APP_FACE_OUTPUT}')

                    methods_df=pd.DataFrame()
                    
                    for single_method in selected_methods:
                        st.write(f"Method: {single_method}")
                        s=time()
                        
                        if   single_method=='Minus Single': image, hm_path,_=app_utils.run_saliency_minus_single(img1_face_path, img2_face_path, img2_face_obj)
                        elif single_method=='Minus Greedy': image, hm_path,_=app_utils.run_saliency_minus_greedy(img1_face_path, img2_face_path, img2_face_obj)
                        elif single_method=='Plus Single':  image, hm_path,_=app_utils.run_saliency_plus_single(img1_face_path, img2_face_path, img2_face_obj)
                        elif single_method=='Plus Greedy':  image, hm_path,_=app_utils.run_saliency_plus_greedy(img1_face_path, img2_face_path, img2_face_obj)
                        elif single_method=='Average':      image, hm_path,_=app_utils.run_saliency_average(img1_face_path, img2_face_path, img2_face_obj)

                        st.image(image, caption=single_method, use_column_width=True)
                        st.session_state[f"{single_model}_{single_method}"] =  image
                        e=time()
                        print(f'{single_method} took {e-s} seconds')

                        s=time()
                        imp_df=app_utils.run_features_importance(uploaded_file1, uploaded_file2, hm_path, method_name=single_method)
                        # imp_df['Model Name']=single_model
                        e=time()
                        print(f'Features Importance took {e-s} seconds')
                        if len(methods_df)==0:
                            methods_df=imp_df.copy()
                        else:
                            methods_df=pd.merge(methods_df, imp_df, on='Landmark Name')
                except Exception as e:
                    st.error(f'Error: {e}')
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print("ERROR: ", e, exc_type, fname, exc_tb.tb_lineno)

                methods_df.insert(0, 'Model Name', single_model)
                methods_df.insert(1, 'Detector Name', single_detector)
                final_imp_df=pd.concat([final_imp_df, methods_df], axis=0)

            s=time()
            final_imp_df = app_utils.get_features_nomination_score(final_imp_df)
            e=time()
            print(f'Features Nomination Score took {e-s} seconds')




            s=time()
            img1_name, img2_name=img1_path.split('/')[-1], img2_path.split('/')[-1]

            #############################################################################################
            # # FOR MAADFACE IMAGES, GET THE TYPES FROM EXCEL FILE
            # if img1_name  in ATTRS_DF['Filename'] and img2_name  in ATTRS_DF['Filename']:   
            #     type_df = app_utils.get_features_type(ATTRS_DF, img1_name, img2_name)
            # # FOR NEW IMAGES, GET THE PREDICTION FROM MODEL
            # else: 
            #     type_df = app_utils.predict_features_type(embeddings, params.ATTR_MODELS[single_model])

            # e=time()
            # print(f'Features Type Prediction took {e-s} seconds')

            # display_df=pd.merge(final_imp_df, type_df, on='Landmark Name')
            #############################################################################################
            
            display_df=final_imp_df.copy()


            display_df=display_df.astype(str)
            st.subheader('All Results')
            st.dataframe(display_df)
            st.session_state['display_df'] = display_df
                
            ########################## SAVING THIS FILE TO BE USED BY CHATBOT QA AS A CONTEXT ##########################
            # display_df.to_csv('output/final_dataframe/final_display_df.csv')
            st.session_state['final_display_df'] = display_df
            ############################################################################################################

            st.subheader('Most Important Features')
            # Convert 'Model Name' column to categorical to ensure correct sorting
            display_df['Model Name'] = pd.Categorical(display_df['Model Name'])
            # Sort the DataFrame based on 'Ratio of 1s' and 'Mean' columns
            best_overall_result=display_df.sort_values(['Model Name', 'Ratio of 1s', 'Mean'], ascending=[True, False, True], inplace=False)
            # Get the rows with maximum 'Ratio of 1s' and minimum 'Mean' for each 'Model Name'
            best_overall_result = best_overall_result.groupby('Model Name').apply(lambda x: x[x['Ratio of 1s'] == x['Ratio of 1s'].max()]).reset_index(drop=True)
            best_overall_result = best_overall_result.groupby('Model Name').apply(lambda x: x[x['Mean'] == x['Mean'].min()]).reset_index(drop=True)
            # best_overall_result = pd.merge(display_df, best_overall_result, on=['Model Name', 'Mean', 'Ratio of 1s'])
            st.table(best_overall_result)
            st.session_state['best_overall_result'] = best_overall_result



            imp={}
            # Iterate through the rows of result_df and construct text explanations
            for model_name, group in best_overall_result.groupby('Model Name'):
                feature_names = ', '.join(group['Landmark Name'].tolist())
                if len(feature_names) == 1:
                    imp[model_name]=feature_names
                else:
                    model_names = ', '.join(group['Model Name'].unique())
                    imp[model_name]=feature_names

            # Create a reverse mapping of values to keys
            reverse_mapping = {}
            for key, value in imp.items():
                if value not in reverse_mapping:
                    reverse_mapping[value] = [key]
                else:
                    reverse_mapping[value].append(key)

            # Construct natural language explanations
            most_imp_explanations = []
            for value, keys in reverse_mapping.items():
                if len(keys) == 1:
                    most_imp_explanations.append(f"For {keys[0]}: {value} is the most important feature.")
                else:
                    model_names = ', '.join(keys)
                    most_imp_explanations.append(f"For {model_names}: {value} are the most important features.")

            st.session_state['most_imp_explanations'] = most_imp_explanations
            # Print or use explanations as needed
            for explanation in most_imp_explanations:
                st.write(f"- {explanation}")


            st.subheader('Least Important Features')
            # Convert 'Model Name' column to categorical to ensure correct sorting
            display_df['Model Name'] = pd.Categorical(display_df['Model Name'])
            # Sort the DataFrame based on 'Ratio of 1s' and 'Mean' columns
            least_overall_result=display_df.sort_values(['Model Name', 'Ratio of 1s', 'Mean'], ascending=[False, True, False], inplace=False)
            # Get the rows with maximum 'Ratio of 1s' and minimum 'Mean' for each 'Model Name'
            least_overall_result = least_overall_result.groupby('Model Name').apply(lambda x: x[x['Ratio of 1s'] == x['Ratio of 1s'].min()]).reset_index(drop=True)
            least_overall_result = least_overall_result.groupby('Model Name').apply(lambda x: x[x['Mean'] == x['Mean'].max()]).reset_index(drop=True)
            # best_overall_result = pd.merge(display_df, best_overall_result, on=['Model Name', 'Mean', 'Ratio of 1s'])
            st.table(least_overall_result)
            st.session_state['least_overall_result'] = least_overall_result

            imp={}
            # Iterate through the rows of result_df and construct text explanations
            for model_name, group in least_overall_result.groupby('Model Name'):
                feature_names = ', '.join(group['Landmark Name'].tolist())
                if len(feature_names) == 1:
                    imp[model_name]=feature_names
                else:
                    model_names = ', '.join(group['Model Name'].unique())
                    imp[model_name]=feature_names

            # Create a reverse mapping of values to keys
            reverse_mapping = {}
            for key, value in imp.items():
                if value not in reverse_mapping:
                    reverse_mapping[value] = [key]
                else:
                    reverse_mapping[value].append(key)

            # Construct natural language explanations
            least_imp_explanations = []
            for value, keys in reverse_mapping.items():
                if len(keys) == 1:
                    least_imp_explanations.append(f"For {keys[0]}: {value} is the least important feature.")
                else:
                    model_names = ', '.join(keys)
                    least_imp_explanations.append(f"For {model_names}: {value} are the least important features.")

            st.session_state['least_imp_explanations'] = least_imp_explanations
            # Print or use explanations as needed
            for explanation in least_imp_explanations:
                st.write(f"- {explanation}")

# Check if a rerun is needed
if 'clear_state' not in st.session_state:
    st.session_state.clear_state = False

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    reset_button_clicked = st.button(
                                    label='🔄 Clear and Re-run App', 
                                    key='reset'
                                )

# Button to clear session state and rerun app
# if st.button('🔄 Clear and Re-run App'):
if reset_button_clicked:
    # Set the flag to clear the session state
    st.session_state.clear_state = True
    # Rerun the app which will now enter the next if block
    st.experimental_rerun()

# If the flag is set, clear the session state
if st.session_state.clear_state:
    # Clear all items in the session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Reset the flag
    st.session_state.clear_state = False
    # Rerun the app to start fresh
    st.experimental_rerun()

