
import pickle, os, pdb
import numpy as np
import pandas as pd
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import cv2 as cv
import time, sys
from facenet_pytorch import MTCNN
from PIL import Image
import skimage
from skimage import io
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm
import shutil



################################################################################ < START GERNERAL > ################################################################################

def dirname(path):
    return os.path.basename(os.path.dirname(path))


def filename(path, extension=True):
    if extension: return os.path.basename(path)
    else: return os.path.basename(os.path.splitext(path)[0])


def recreate_folder(path, subdirs=[]):
    try: shutil.rmtree(path)
    except: pass
    os.mkdir(path)
    if subdirs:
        for x in subdirs:
            os.mkdir(f"{path}/{x}")


def typingPrint(text):
  for character in text:
    sys.stdout.write(character)
    sys.stdout.flush()
    time.sleep(0.05)


def clearScreen():
  os.system("clear")


def save_as_pickle(fname, data):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(fname):
    with open(fname, 'rb') as handle:
        b = pickle.load(handle)
    return b


def calculate_cosine_similarity(df, static=['image_id','identity']):
    # Extract the feature columns
    feature_columns = df.drop(columns=static)  # Adjust if needed

    # Calculate the cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(feature_columns)

    # Create a DataFrame from the similarity matrix
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=df.image_id, columns=df.image_id)

    return cosine_sim_df, cosine_sim_matrix


def get_bulls_eye_score(df, cosine_sim_matrix, top_k=30, num_classes = 10,samples_per_class = 10):

    # Initialize the Bulls Eye Score
    bulls_eye_score = 0

    ground_truth_classes=df.identity.tolist()

    # Iterate through each shape
    for i in range(len(ground_truth_classes)):
        # Get the indices of the most similar shapes for the current shape
        most_similar_indices = np.argsort(cosine_sim_matrix[i])[::-1][:top_k]

        # Get the classes of the most similar shapes
        most_similar_classes = [ground_truth_classes[idx] for idx in most_similar_indices]
        
        # Count the number of shapes from the same class among the most similar shapes
        same_class_count = most_similar_classes.count(ground_truth_classes[i])
        
        # Update the Bulls Eye Score
        bulls_eye_score += same_class_count

    # Calculate the total number of possible correct retrievals
    total_possible_retrievals = samples_per_class * (num_classes * samples_per_class)

    # Calculate the Bulls Eye Score
    bulls_eye_score /= total_possible_retrievals
    bulls_eye_score *= 100

    return bulls_eye_score


def print_images(img_names):
    for img_name in img_names:
        # print(img_name)

        plt.figure()

        im_cv = cv.imread(img_name)[:, :, ::-1] 
        plt.axis("on")

        plt.imshow(im_cv) 
        plt.show()

################################################################################ </ END GERNERAL> ################################################################################




################################################################################# < START FACENET> ################################################################################
def get_face_embds_facenet_single(path):
    image=Image.open(path)

    mtcnn = MTCNN()
    face = mtcnn(image)

    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    img_embedding = resnet(face.unsqueeze(0)).detach().numpy()
    return img_embedding

def detect_and_crop_face(img_obj, resize=(256,256)):
    try:
        mtcnn=MTCNN(image_size=resize[0])
        face = mtcnn(img_obj)

        return face
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname, exc_tb.tb_lineno)


def get_face_embds_facenet_inference(img1_obj, img2_obj, model_name, resize=(256,256)):
    try:
        images=[img1_obj.resize(resize).convert('RGB'), img2_obj.resize(resize).convert('RGB')]

        mtcnn=MTCNN(image_size=resize[0])
        faces = mtcnn(images)
        faces_stacked=torch.stack(faces)

        resnet = InceptionResnetV1(pretrained=model_name).eval()
        embeddings = resnet(faces_stacked).detach().numpy()

        return embeddings

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(e, exc_type, fname, exc_tb.tb_lineno)


def get_face_embds_facenet_multi(paths,cropped_save_path=None,margin=56, select_largest=False, post_process=False, resize=(256,256), save_path=None):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    if resize:
        mtcnn=MTCNN(image_size=resize[0])

        embeddings=[]
        processed_paths=[]
        faces=[]
        finaldf=pd.DataFrame()
        i=0
        for path in tqdm(paths):
            try:

                img=Image.open(path).resize(resize)

                if cropped_save_path:
                    face = mtcnn(img, save_path=f"{cropped_save_path}/{dirname(path)}/{filename(path)}")
                else:
                    face = mtcnn(img)


                # bacth embeddings can cause errors, so doing in loop
                face_embedding = resnet(face.unsqueeze(0)).detach().numpy()
                embeddings.append(list(face_embedding[0]))
                processed_paths.append(path)
                i+=1
            except Exception as e:
                # exc_type, exc_obj, exc_tb = sys.exc_info()
                # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(e,exc_type, fname, exc_tb.tb_lineno)
                pass
            
            if i%50==0:
                tmpdf=pd.DataFrame({
                        "Filename": [f"{dirname(path)}/{filename(path)}" for path in processed_paths], 
                        "embeddings": embeddings
                    })
                finaldf=pd.concat([finaldf, tmpdf], axis=0)
                finaldf.to_csv(f"{save_path}/sample_10000_embeddings.csv", index=False)

        edf=pd.DataFrame({
                    "Filename": [f"{dirname(path)}/{filename(path)}" for path in processed_paths], 
                    "embeddings": embeddings
                })
        return edf
        
    else:
        images=[Image.open(path) for path in tqdm(paths)]
        mtcnn = MTCNN()
        faces = mtcnn(images)
    
        faces_stacked=torch.stack(faces)

        embeddings = resnet(faces_stacked).detach().numpy()

        return embeddings


################################################################################# </ END FACENET> ################################################################################





