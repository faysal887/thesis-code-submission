import numpy as np
import pandas as pd
from PIL import Image
from time import time
import streamlit as st
from stqdm import stqdm
from deepface import DeepFace
import pdb, dlib, sys, os, cv2
from sklearn.metrics.pairwise import cosine_similarity

import utils.params as params
import utils.face_landmarks_utils as lm_utils
import utils.helper_functions as helper_functions
import utils.blackbox_explaination.fvx_utils as fxv
import utils.blackbox_explaination.fvx_utils_helper as fxv_2


def get_save_fn(img_A, img_B):
    return f"{'_'.join([os.path.basename(img_A),os.path.basename(img_B)]).replace('.png','').replace('.jpg','').replace('.jpeg','')}.png"


def get_input_images(img1_path, img2_path, image2):
        # img_A=f"{APP_UPLOADED}/{uploaded_file1.name}"
        # img_B=f"{APP_UPLOADED}/{uploaded_file2.name}"

        save_fn=get_save_fn(img1_path, img2_path)

        image1 = Image.open(img1_path).convert('RGB')
        image2 = Image.open(img2_path).convert('RGB')

        A=np.array(image1.resize((256,256)))
        Bo=np.array(image2.resize((256,256)))
        (N,M)     = A.shape[0:2]
        As        = A.copy()
        Bs        = Bo.copy()
        return A, Bo, As, Bs, save_fn


def features_imp(landmarks_dict, heatmap_image, method_name):
    # heatmap_image=f'{params.APP_OUTPUT}/HM_{save_fn}'
    lm_imp={}
    for lm, points in landmarks_dict.items():
        try:
            # LM='chin'
            if lm=='jawline': continue

            imp=lm_utils.get_dominant_importance_level(heatmap_image, points, display=True, crop=True)
            lm_imp[lm]=imp
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f"\nERROR {lm}: {e}\n")

    # imp_df=pd.DataFrame(lm_imp).sort_values(by=0, axis=1)
    imp_df=pd.DataFrame({'Landmark Name': list(lm_imp.keys()), f'{method_name}': list(lm_imp.values())})
    # imp_df=imp_df.sort_values(['Importance']).reset_index(drop=True)
    return imp_df


def predict_features_type(embeddings, ATTR_MODEL):
    cols=[
        'gender', 'face_type', 'cheeks_type', 'skin_type', 'cheekbones_type', 
        'skin_under_eyes', 'mouth_position', 'lips_type', 'jawline_type',
        'eye_color', 'beard_type', 'eyewear_type', 'makeup_type', 'age', 
        'race', 'face_shape', 'forehead', 'eyebrow', 'nose',
    ]

    # image_embedding_2 = utils.get_face_embds_facenet_inference(f"{APP_UPLOADED}/{image2}",resize=(256,256))
    image_embedding_1 =  np.array(embeddings[0]).reshape(1, -1)
    image_embedding_2 =  np.array(embeddings[1]).reshape(1, -1)

    # type_df=pd.DataFrame({
    #     'Landmark Name':['right_cheek', 'left_cheek', 'left_eye', 'right_eye', 'right_eyebrow', 'left_eyebrow', 'chin', 'nose', 'lips'],
    #     # 'Type':['High Cheekbones', 'High Cheekbones', 'Brown Eye', 'Brown Eye', 'High Eyebrows', 'High Eyebrows', 'Double Chin', 'Big Nose', 'Narrow Lips']
    # })

    pred1=ATTR_MODEL.predict(np.asarray(image_embedding_1).reshape(1, -1))
    pred2=ATTR_MODEL.predict(np.asarray(image_embedding_2).reshape(1, -1))

    pred_df=pd.DataFrame({'Landmark Name':cols, 'Type Photo-1': pred1[0], 'Type Photo-2': pred2[0]})

    # converting boolean values of eyewear_type into 'have eyewear' and 'no eyewear'
    pred_df.loc[pred_df['Landmark Name'] == 'eyewear_type', 'Type Photo-1'] = pred_df.loc[pred_df['Landmark Name'] == 'eyewear_type', 'Type Photo-1'].replace({'True': 'have eyewear', 'False': 'no eyewear'})
    pred_df.loc[pred_df['Landmark Name'] == 'eyewear_type', 'Type Photo-2'] = pred_df.loc[pred_df['Landmark Name'] == 'eyewear_type', 'Type Photo-2'].replace({'True': 'have eyewear', 'False': 'no eyewear'})

    def get(feature_name, photo_n):
        return pred_df[pred_df['Landmark Name']==feature_name][f'Type Photo-{photo_n}'].item()

    if get('gender',1)=='male':
        right_cheek_1 = get('skin_type',1) +' with '+ get('face_type',1) +' '+ get('cheeks_type',1) +' and '+ get('cheekbones_type',1) +' and have '+ get('beard_type',1)
        left_cheek_1  = get('skin_type',1) +' with '+ get('face_type',1) +' '+ get('cheeks_type',1) +' and '+ get('cheekbones_type',1) +' and have '+ get('beard_type',1)
    if get('gender',1)=='female':
        right_cheek_1 = get('skin_type',1) +' with '+ get('face_type',1) +' '+ get('cheeks_type',1) +' and '+ get('cheekbones_type',1) +' and wearing '+ get('makeup_type',1),
        left_cheek_1  = get('skin_type',1) +' with '+ get('face_type',1) +' '+ get('cheeks_type',1) +' and '+ get('cheekbones_type',1) +' and wearing '+ get('makeup_type',1),
    if get('gender',2)=='male':
        right_cheek_2 = get('skin_type',2) +' with '+ get('face_type',2) +' '+ get('cheeks_type',2) +' and '+ get('cheekbones_type',2) +' and have '+ get('beard_type',2)
        left_cheek_2  = get('skin_type',2) +' with '+ get('face_type',2) +' '+ get('cheeks_type',2) +' and '+ get('cheekbones_type',2) +' and have '+ get('beard_type',2)
    if get('gender',2)=='female':
        right_cheek_2 = get('skin_type',2) +' with '+ get('face_type',2) +' '+ get('cheeks_type',2) +' and '+ get('cheekbones_type',2) +' and wearing '+ get('makeup_type',2),
        left_cheek_2  = get('skin_type',2) +' with '+ get('face_type',2) +' '+ get('cheeks_type',2) +' and '+ get('cheekbones_type',2) +' and wearing '+ get('makeup_type',2),

    desc_dict1={
    'right_cheek':   right_cheek_1[0],
    'left_cheek':    left_cheek_1[0],
    'left_eye':      get('eye_color',1)   +' with '+ get('eyewear_type',1) +' and '+ get('skin_under_eyes',1),
    'right_eye':     get('eye_color',1)   +' with '+ get('eyewear_type',1) +' and '+ get('skin_under_eyes',1),
    'right_eyebrow': get('eyebrow',1),
    'left_eyebrow':  get('eyebrow',1),
    'chin':          get('jawline_type',1),
    'nose':          get('nose',1),
    'lips':          get('lips_type',1) +' - '+ get('mouth_position',1),
    }
    desc_dict2={
    'right_cheek':   right_cheek_2[0],
    'left_cheek':    left_cheek_2[0],
    'left_eye':      get('eye_color',2)   +' with '+ get('eyewear_type',2) +' and '+ get('skin_under_eyes',2),
    'right_eye':     get('eye_color',2)   +' with '+ get('eyewear_type',2) +' and '+ get('skin_under_eyes',2),
    'right_eyebrow': get('eyebrow',2),
    'left_eyebrow':  get('eyebrow',2),
    'chin':          get('jawline_type',2),
    'nose':          get('nose',2),
    'lips':          get('lips_type',2) +' - '+ get('mouth_position',2),
    }
    desc_df1 = pd.DataFrame(desc_dict1.items(), columns=['Landmark Name', 'Feature Explanation (Photo-1)'])
    desc_df2 = pd.DataFrame(desc_dict2.items(), columns=['Landmark Name', 'Feature Explanation (Photo-2)'])

    desc_df=pd.merge(desc_df1, desc_df2, on='Landmark Name')
    return desc_df  


def get_features_type(attrs_df, img1_name, img2_name):
    cols=[
        'gender', 'face_type', 'cheeks_type', 'skin_type', 'cheekbones_type', 
        'skin_under_eyes', 'mouth_position', 'lips_type', 'jawline_type',
        'eye_color', 'beard_type', 'eyewear_type', 'makeup_type', 'age', 
        'race', 'face_shape', 'forehead', 'eyebrow', 'nose',
    ]

    pred1 = attrs_df[attrs_df.Demo_Filename == img1_name][cols].values
    pred2 = attrs_df[attrs_df.Demo_Filename == img2_name][cols].values
    pred_df=pd.DataFrame({'Landmark Name':cols, 'Type Photo-1': pred1[0], 'Type Photo-2': pred2[0]})

    # converting boolean values of eyewear_type into 'have eyewear' and 'no eyewear'
    pred_df.loc[pred_df['Landmark Name'] == 'eyewear_type', 'Type Photo-1'] = pred_df.loc[pred_df['Landmark Name'] == 'eyewear_type', 'Type Photo-1'].replace({'True': 'have eyewear', 'False': 'no eyewear'})
    pred_df.loc[pred_df['Landmark Name'] == 'eyewear_type', 'Type Photo-2'] = pred_df.loc[pred_df['Landmark Name'] == 'eyewear_type', 'Type Photo-2'].replace({'True': 'have eyewear', 'False': 'no eyewear'})

    def get(feature_name, photo_n):
        return pred_df[pred_df['Landmark Name']==feature_name][f'Type Photo-{photo_n}'].item()

    if get('gender',1)=='male':
        right_cheek_1 = get('skin_type',1) +' with '+ get('face_type',1) +' '+ get('cheeks_type',1) +' and '+ get('cheekbones_type',1) +' and have '+ get('beard_type',1)
        left_cheek_1  = get('skin_type',1) +' with '+ get('face_type',1) +' '+ get('cheeks_type',1) +' and '+ get('cheekbones_type',1) +' and have '+ get('beard_type',1)
    if get('gender',1)=='female':
        right_cheek_1 = get('skin_type',1) +' with '+ get('face_type',1) +' '+ get('cheeks_type',1) +' and '+ get('cheekbones_type',1) +' and wearing '+ get('makeup_type',1),
        left_cheek_1  = get('skin_type',1) +' with '+ get('face_type',1) +' '+ get('cheeks_type',1) +' and '+ get('cheekbones_type',1) +' and wearing '+ get('makeup_type',1),
    if get('gender',2)=='male':
        right_cheek_2 = get('skin_type',2) +' with '+ get('face_type',2) +' '+ get('cheeks_type',2) +' and '+ get('cheekbones_type',2) +' and have '+ get('beard_type',2)
        left_cheek_2  = get('skin_type',2) +' with '+ get('face_type',2) +' '+ get('cheeks_type',2) +' and '+ get('cheekbones_type',2) +' and have '+ get('beard_type',2)
    if get('gender',2)=='female':
        right_cheek_2 = get('skin_type',2) +' with '+ get('face_type',2) +' '+ get('cheeks_type',2) +' and '+ get('cheekbones_type',2) +' and wearing '+ get('makeup_type',2),
        left_cheek_2  = get('skin_type',2) +' with '+ get('face_type',2) +' '+ get('cheeks_type',2) +' and '+ get('cheekbones_type',2) +' and wearing '+ get('makeup_type',2),

    desc_dict1={
    'right_cheek':   right_cheek_1[0],
    'left_cheek':    left_cheek_1[0],
    'left_eye':      get('eye_color',1)   +' with '+ get('eyewear_type',1) +' and '+ get('skin_under_eyes',1),
    'right_eye':     get('eye_color',1)   +' with '+ get('eyewear_type',1) +' and '+ get('skin_under_eyes',1),
    'right_eyebrow': get('eyebrow',1),
    'left_eyebrow':  get('eyebrow',1),
    'chin':          get('jawline_type',1),
    'nose':          get('nose',1),
    'lips':          get('lips_type',1) +' - '+ get('mouth_position',1),
    }
    desc_dict2={
    'right_cheek':   right_cheek_2[0],
    'left_cheek':    left_cheek_2[0],
    'left_eye':      get('eye_color',2)   +' with '+ get('eyewear_type',2) +' and '+ get('skin_under_eyes',2),
    'right_eye':     get('eye_color',2)   +' with '+ get('eyewear_type',2) +' and '+ get('skin_under_eyes',2),
    'right_eyebrow': get('eyebrow',2),
    'left_eyebrow':  get('eyebrow',2),
    'chin':          get('jawline_type',2),
    'nose':          get('nose',2),
    'lips':          get('lips_type',2) +' - '+ get('mouth_position',2),
    }
    desc_df1 = pd.DataFrame(desc_dict1.items(), columns=['Landmark Name', 'Feature Explanation (Photo-1)'])
    desc_df2 = pd.DataFrame(desc_dict2.items(), columns=['Landmark Name', 'Feature Explanation (Photo-2)'])

    desc_df=pd.merge(desc_df1, desc_df2, on='Landmark Name')
    return desc_df  


def run_features_importance(uploaded_file1,uploaded_file2,hm_path, method_name):
        save_fn=get_save_fn(uploaded_file1.name, uploaded_file2.name)
        
        face_image=f'{params.APP_UPLOADED}/{uploaded_file2.name}'
        landmarks_dict, lm=lm_utils.detect_landmarks_dlib(
                                        face_image, 
                                        params.DLIB_FACE_DETECT_MODEL,
                                        params.DLIB_LM_MODEL, 
                                        display=False, 
                                        save=True, 
                                        save_path=f"{params.APP_OUTPUT}/landmarks_{save_fn}", 
                                        resize=(256, 256),
                            )
        # landmarks_img=Image.open(f"{params.APP_OUTPUT}/landmarks_{save_fn}").convert('RGB')
        # left_co, cent_co,last_co = st.columns(3)
        # with cent_co: st.image(landmarks_img, caption='Landmarks Detected', use_column_width=True)
        

        imp_df=features_imp(landmarks_dict,hm_path, method_name)

        # finaldf.Importance=finaldf.Importance.astype(str)
        # imp_df=imp_df.astype(str)

        return imp_df


def crop_and_save_face(img_path, model_name, detector_name, cropped_face_dir):
    try:
        x = DeepFace.represent(
                        img_path, 
                        model_name=model_name, 
                        detector_backend=detector_name, 
                        enforce_detection=False, 
                        align=True,
            )
        _=x[0]["embedding"]
        bbox=x[0]["facial_area"]
        image = Image.open(img_path)

        # Get the bounding box coordinates
        x = bbox['x']
        y = bbox['y']
        w = bbox['w']
        h = bbox['h']

        # Crop the region defined by the bounding box
        cropped_image = image.crop((x, y, x + w, y + h))
        save_path=f'{cropped_face_dir}/{img_path.split("/")[-1]}'
        cropped_image.save(save_path)

        return cropped_image, save_path
    
    except Exception as e:
        print('Error in app_utils/crop_and_save_face: ', e)


def run_comparison_result(image1,image2, model_name):
    try:

        if model_name in ['vggface2','casia-webface']:
            embeddings = helper_functions.get_face_embds_facenet_inference(image1,image2,model_name, resize=(256,256))
            face_similarity = cosine_similarity(embeddings[0].reshape(1, -1),embeddings[1].reshape(1, -1))
        
        if model_name in ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepID", "ArcFace"]:
            e1=fxv.face_embedding(np.array(image1))
            e2=fxv.face_embedding(np.array(image2))
            face_similarity = cosine_similarity([e1], [e2])

            embeddings=[e1,e2]

        
        # if face_similarity >= FACE_THRESHOLD:
        #     st.success(f'These are the photos of same person - {face_similarity}')
        # else:
        #     st.error(f'These are the photos of different persons - {face_similarity}')

        return embeddings
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('Error in app_utils/run_comparison_result: ', e, exc_type, fname, exc_tb.tb_lineno)


def run_saliency_minus_single(uploaded_file1, uploaded_file2, image2, display=False):
    s=time()
    A, Bo, As,Bs,save_fn=get_input_images(uploaded_file1, uploaded_file2, image2)
    e=time()
    print(f'get_input_images took {e-s} seconds')
    S0m, HM1_m =    fxv_2.get_sliency_minus_single(
                                A, Bo, As,Bs, 
                                display=True, 
                                contour_face_save_path=f'{params.APP_OUTPUT}/contour_face_minus_single_{save_fn}',
                                contour_no_face_save_path=f'{params.APP_OUTPUT}/contour_no_face_minus_single_{save_fn}',
                                save_path=f'{params.APP_OUTPUT}/merged_minus_{save_fn}'
                    )
    image = Image.open(f'{params.APP_OUTPUT}/merged_minus_{save_fn}')
    # if display: st.image(image, caption='Saliency Minus', use_column_width=True)

    hm_path=f'{params.APP_OUTPUT}/HM_S0_MINUS_{save_fn}'
    cv2.imwrite(hm_path, HM1_m)
    return image, hm_path, S0m


def run_saliency_minus_greedy(uploaded_file1, uploaded_file2, image2, display=False):
    A, Bo, As,Bs,save_fn=get_input_images(uploaded_file1, uploaded_file2, image2)
    S1m, HM2_m = fxv_2.get_sliency_minus_greedy(
                                A, Bo, As,Bs, 
                                display=True, 
                                save_path=f'{params.APP_OUTPUT}/merged_minus_{save_fn}',
                                contour_face_save_path=f'{params.APP_OUTPUT}/contour_face_minus_greedy_{save_fn}',
                                contour_no_face_save_path=f'{params.APP_OUTPUT}/contour_no_face_minus_greedy_{save_fn}',

                )
    image = Image.open(f'{params.APP_OUTPUT}/merged_minus_{save_fn}')
    # if display: st.image(image, caption='Saliency Minus', use_column_width=True)

    hm_path=f'{params.APP_OUTPUT}/HM_S1_MINUS_{save_fn}'
    cv2.imwrite(hm_path, HM2_m)
    return image, hm_path, S1m


def run_saliency_plus_single(uploaded_file1, uploaded_file2, image2, display=False):
    A, Bo, As,Bs,save_fn=get_input_images(uploaded_file1, uploaded_file2, image2)
    S0p, HM1_p = fxv_2.get_sliency_plus_single(
                        A,  Bo, As,Bs, 
                        display=True, 
                        save_path=f'{params.APP_OUTPUT}/merged_plus_{save_fn}',
                        contour_face_save_path=f'{params.APP_OUTPUT}/contour_face_plus_single_{save_fn}',
                        contour_no_face_save_path=f'{params.APP_OUTPUT}/contour_no_face_plus_single_{save_fn}',
                )
    image = Image.open(f'{params.APP_OUTPUT}/merged_plus_{save_fn}')
    # if display: st.image(image, caption='Saliency Plus', use_column_width=True)

    hm_path=f'{params.APP_OUTPUT}/HM_S0_PLUS_{save_fn}'
    cv2.imwrite(hm_path, HM1_p)
    return image, hm_path, S0p


def run_saliency_plus_greedy(uploaded_file1, uploaded_file2, image2, display=False):
    A, Bo, As,Bs,save_fn=get_input_images(uploaded_file1, uploaded_file2, image2)
    S1p,HM2_p = fxv_2.get_sliency_plus_greedy(
                        A,  Bo, As,Bs, 
                        display=True, 
                        save_path=f'{params.APP_OUTPUT}/merged_plus_{save_fn}',
                        contour_face_save_path=f'{params.APP_OUTPUT}/contour_face_plus_greedy_{save_fn}',
                        contour_no_face_save_path=f'{params.APP_OUTPUT}/contour_no_face_plus_greedy_{save_fn}',
                )
    image = Image.open(f'{params.APP_OUTPUT}/merged_plus_{save_fn}')
    # if display: st.image(image, caption='Saliency Plus', use_column_width=True)

    hm_path=f'{params.APP_OUTPUT}/HM_S1_PLUS_{save_fn}'
    cv2.imwrite(hm_path, HM2_p)
    return image, hm_path, S1p


def run_saliency_average(uploaded_file1, uploaded_file2, image2):
    try:
        A, Bo, As,Bs,save_fn=get_input_images(uploaded_file1, uploaded_file2, image2)

        _,_, S0m=run_saliency_minus_single(uploaded_file1, uploaded_file2, image2, display=False)
        _,_, S1m=run_saliency_minus_greedy(uploaded_file1, uploaded_file2, image2, display=False)
        _,_, S0p=run_saliency_plus_single(uploaded_file1, uploaded_file2, image2, display=False)
        _,_, S1p=run_saliency_plus_greedy(uploaded_file1, uploaded_file2, image2, display=False)

        HM            = fxv_2.get_sliency_avg(S0m, S0p,S1m,S1p, 
                                        A,  Bo, As,Bs,
                                        display=True,
                                        merged_save_path=f'{params.APP_OUTPUT}/merged_avg_{save_fn}',
                                        face_heatmap_save_path=f'{params.APP_OUTPUT}/merged_avg_heatmap_{save_fn}', # save face heatmap of probe image
                                        contour_face_save_path=f'{params.APP_OUTPUT}/contour_face_average_{save_fn}',
                                        contour_no_face_save_path=f'{params.APP_OUTPUT}/contour_no_face_average_{save_fn}',
                                )
        hm_path=f'{params.APP_OUTPUT}/HM_AVG_{save_fn}'
        cv2.imwrite(hm_path, HM)

        image = Image.open(f'{params.APP_OUTPUT}/merged_avg_{save_fn}')
        # st.image(image, caption='Saliency Average', use_column_width=True)
        return image, hm_path, _
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("ERROR in run_saliency_average: ", e, exc_type, fname, exc_tb.tb_lineno)
        return


def run_saliency_seq(uploaded_file1, uploaded_file2, image2):
    A, Bo, As,Bs,save_fn=get_input_images(uploaded_file1, uploaded_file2, image2)
    HM           = fxv_2.get_saliency_seq(
                                A,  Bo, As,Bs,
                                merged_save_path=f'{params.APP_OUTPUT}/merged_seq_{save_fn}',
                                display=True,
                        )
    hm_path=f'{params.APP_OUTPUT}/HM_SEQ_{save_fn}'
    cv2.imwrite(hm_path, HM)

    image = Image.open(f'{params.APP_OUTPUT}/merged_seq_{save_fn}')
    st.image(image, caption='Saliency SEQ', use_column_width=True)
    return hm_path


def run_saliency_lime(uploaded_file1, uploaded_file2, image2):
    A, Bo, As,Bs,save_fn=get_input_images(uploaded_file1, uploaded_file2, image2)

    HM           = fxv_2.get_saliency_lime(
                        A,  Bo, As,Bs,
                        merged_save_path=f'{params.APP_OUTPUT}/merged_lime_{save_fn}',
                        display=True,
                        N=500,
                )
    hm_path=f'{params.APP_OUTPUT}/HM_LIME_{save_fn}'
    cv2.imwrite(hm_path, HM)

    image = Image.open(f'{params.APP_OUTPUT}/merged_lime_{save_fn}')
    st.image(image, caption='Saliency LIME', use_column_width=True)
    return hm_path


def run_saliency_rise(uploaded_file1, uploaded_file2, image2, kernel):
    A, Bo, As,Bs,save_fn=get_input_images(uploaded_file1, uploaded_file2, image2)
    if kernel == 'gauss':
        HM           = fxv_2.get_saliency_rise_gauss(
                        A,  Bo, As,Bs,
                        merged_save_path=f'{params.APP_OUTPUT}/merged_rise_{kernel}_{save_fn}',
                        display=True,
                        contour_face_save_path=f'{params.APP_OUTPUT}/contour_face_rise_gauss_{save_fn}',
                        contour_no_face_save_path=f'{params.APP_OUTPUT}/contour_no_face_rise_gauss_{save_fn}',
                )

    if kernel == 'square':
        HM           = fxv_2.get_saliency_rise_square(
                        A,  Bo, As,Bs,
                        merged_save_path=f'{params.APP_OUTPUT}/merged_rise_{kernel}_{save_fn}',
                        display=True,
                        contour_face_save_path=f'{params.APP_OUTPUT}/contour_face_rise_square_{save_fn}',
                        contour_no_face_save_path=f'{params.APP_OUTPUT}/contour_no_face_rise_square_{save_fn}',
                )

    hm_path=f'{params.APP_OUTPUT}/HM_RISE_{kernel}_{save_fn}'
    cv2.imwrite(hm_path, HM)

    image = Image.open(f'{params.APP_OUTPUT}/merged_rise_{kernel}_{save_fn}')
    st.image(image, caption=f'RISE {kernel.upper()}', use_column_width=True)
    return hm_path


def get_features_nomination_score(df):

    # Define a custom function to calculate the mean for integer values
    def calculate_mean(row):
        # Filter the row to include only integer values
        int_values = [x for x in row if isinstance(x, int)]
        
        # Calculate the mean if there are integer values, otherwise return NaN
        if int_values:
            return sum(int_values) / len(int_values)
        else:
            return float('nan')
        
        # Define a custom function to count the number of columns with value 1
    def count_ones(row):
        return sum([1 for x in row if x == 1])


    # Count the total number of integer columns
    total_int_columns = df.applymap(lambda x: isinstance(x, int)).sum(axis=1)

    # Count the number of columns with value 1 for each row
    df['Ones_Count'] = df.apply(count_ones, axis=1)

    # Apply the custom function to each row and create a new column 'Mean'
    df['Mean'] = df.drop(columns=['Ones_Count']).apply(calculate_mean, axis=1).round(2)

    # Calculate the ratio of Ones_Count to total_int_columns and store it in a new column 'Ratio'
    df['Ratio of 1s'] = df['Ones_Count'] / total_int_columns
    df['Ratio of 1s'] = df['Ratio of 1s'].round(2)

    # Print the updated DataFrame
    return df


def run_face_comparison(img1_path, img2_path, label, models, metrics, backends):

    r=[]

    for model in stqdm(models):
        for distance_metric in metrics:
            for backend in backends:
                try:
                    # result = DeepFace.verify(
                    #                     img1_path = img1_path, 
                    #                     img2_path = img2_path, 
                    #                     distance_metric = distance_metric,
                    #                     model_name = model,
                    #                     detector_backend=backend,
                    #                     enforce_detection=False
                    #                 )
                    # r.append(result)
                    embedding_1 = DeepFace.represent(
                                                img_path = img1_path, 
                                                enforce_detection=False,
                                                model_name = model,
                                                detector_backend=backend,
                                            )[0]["embedding"]
                    embedding_2 = DeepFace.represent(
                                                img_path = img2_path, 
                                                enforce_detection=False,
                                                model_name = model,
                                                detector_backend=backend,
                                            )[0]["embedding"]

                    similar = round(cosine_similarity([embedding_1], [embedding_2])[0][0], 2)
                    r.append({'model': model, 'similarity': similar, 'similarity_metric': 'cosine', 'detector_backend': backend})

                except Exception as e:
                    print([model, backend, distance_metric, e])

    # pdb.set_trace()

    rdf=pd.DataFrame(r)#.sort_values('verified', ascending=False)

    # deepface lib already provides the threshold column in its results, so we have to replace it with our own calculated threhsolds
    # open_face=0.35, acc=0.85, fmr=0.10
    # arcface=0.20, acc=0.81, fmr=0.02
    # vggaface2=0.45, acc=0.90, fmr=0.05
    custom_thresholds={
                # 'VGG-Face': 0.45, 
                # 'OpenFace': 0.35, 
                'ArcFace' : params.THRESHOLDS['ArcFace'],  
    }

    # rdf['similarity_threshold']=1-rdf['threshold']
    rdf['similarity_threshold'] = rdf['model'].map(custom_thresholds)
    rdf['verified'] = 'True' if sum(rdf['similarity'] >= rdf['similarity_threshold']) else 'False'
    # rdf['similarity']=round(1-rdf['distance'], 2)
    # rdf['similarity_margin'] = round(rdf['similarity']-rdf['similarity_threshold'], 2)
    rdf['label']=label
    rdf = rdf[['model', 'detector_backend', 'similarity_metric', 'similarity_threshold', 'similarity', 'verified', 'label']]# faical_areas, time

    return rdf


def display_comparison_result(rdf):
    myresult=rdf.verified.value_counts(normalize=True).reset_index()
    i=myresult.proportion.idxmax()
    is_match=myresult.loc[i].verified
    conf=int(round(myresult.loc[i].proportion, 2)*100)

    if is_match:
        st.success(f"{len(rdf[rdf.verified.isin([True, 'True'])])}/{len(rdf)}({conf}%) models validated the pairs")
    else:
        st.error(f"{len(rdf[rdf.verified.isin([True, 'True'])])}/{len(rdf)}({conf}%) models validated the pairs")
######### funcs ##############