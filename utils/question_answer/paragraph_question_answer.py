from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch, pdb, re, os
import torch.nn.functional as F
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import utils.params  as params
from torch.nn.functional import softmax
from models import *

# question answering model using pipeline
# question_answerer = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad', cache_dir='/data/faysal/thesis/models')
# result = question_answerer(question=query, context=context)


# Step 1: Load the tokenizer and model
qa_tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', cache_dir='./models')
qa_model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', cache_dir='./models')


# sentence embeddings model
embeddings_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2', cache_dir='./models')
embeddings_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2', cache_dir='./models')


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embeddings(sentences):
    # Tokenize sentences
    encoded_input = embeddings_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = embeddings_model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.numpy()


def get_most_imp_landmarks(df):
    min_mean = df[df.Mean == df.Mean.min()]
    max_ratio = min_mean[min_mean['Ratio of 1s'] == min_mean['Ratio of 1s'].max()]
    landmarks = max_ratio['Landmark Name'].tolist()
    mean = max_ratio['Mean'].unique().tolist()
    ratio = max_ratio['Ratio of 1s'].unique().tolist()
    return landmarks, mean, ratio


def get_least_imp_landmarks(df):
    max_mean = df[df.Mean == df.Mean.max()]
    min_ratio = max_mean[max_mean['Ratio of 1s'] == max_mean['Ratio of 1s'].min()]
    landmarks = min_ratio['Landmark Name'].tolist()
    mean = min_ratio['Mean'].unique().tolist()
    ratio = min_ratio['Ratio of 1s'].unique().tolist()
    return landmarks, mean, ratio


def get_landmark_type(df, landmark_name):
    type_1 = df[df['Landmark Name']==landmark_name][f'Feature Explanation (Photo-1)'].item()
    type_2 = df[df['Landmark Name']==landmark_name][f'Feature Explanation (Photo-1)'].item()
    return type_1, type_2


def get_score(df, landmark_name, algo_name):
    if algo_name in df.columns:
        score = df[df['Landmark Name']==landmark_name][algo_name].item()
    else:
        score = 'not available'
    return score



def get_landmark_importance(landmark, df):
    # chin:
    # if landmark in imp_landmarks:
    #     text += f" Chin is one of the most important feature." 
    # elif landmark in least_imp_landmarks:
    #     text += f" {landmark} is one of the least important feature."
    # else:
    #     text += f" {landmark} has moderate importance."

    text  = ""
    text += f" Minus single score for {landmark} is {get_score(df, landmark, 'Minus Single')}."
    text += f" Minus greedy score for {landmark} is {get_score(df, landmark, 'Minus Greedy')}."
    text += f" Plus single score for {landmark} is {get_score(df, landmark, 'Plus Single')}."
    text += f" Plus greedy score for {landmark} is {get_score(df, landmark, 'Plus Greedy')}." 
    text += f" Average score for {landmark} is {get_score(df, landmark, 'Average')}."
    text += f" Mean value of {landmark} is {get_score(df, landmark, 'Mean')}."
    text += f" Ratio of 1s for {landmark} is {get_score(df, landmark,'Ratio of 1s')}."
    text += "\n"

    return text


# Function to convert the dataframe into a detailed text paragraph
def dataframe_to_text(df, binary_df):

    binary_decision = 'genuine' if binary_df.verified.item()=='True' else 'imporster'
    imp_landmarks, imp_mean, imp_ratio = get_most_imp_landmarks(df)
    least_imp_landmarks, least_imp_mean, least_imp_ratio = get_least_imp_landmarks(df)
    pic_score_confidence = round(binary_df.decision_confidence.item(), 2)*100
    cosine_similarity=binary_df.similarity.item()*100
    avl_approaches = params.XAI_APPROACHES
    selected_approaches = set(df.columns.tolist()).intersection(set(avl_approaches))

    full_context=f"""
    The decision is that the face images pair is {binary_decision}. \n
    Genuine pair means that these are the photos of the same person. \n
    An imposter pair means that these are the photos of different persons. \n
    The cosine similarity of the given pair is {cosine_similarity}%. \n
    The minimum similarity of 31%' is required for a pair to be called genuine. \n

    I come to this decision by calculating the cosine similarity on the face templates generated by ArcFace model. \n
    ArcFace is a face recognition model which used in our pipeline. \n
    Face templates are the vector values of the face images generated by ArcFace model. \n

    We are {pic_score_confidence}% confident about our decision. \n
    The confidence is calulcated by PIC-score. \n
    A PIC-score is the probability that the comparison belongs to a genuine (same person) comparison. \n
    Learn about PIC-score here https://github.com/pterhoer/optimalmatchingconfidence. \n

    The cosine similarity ranges from -1 to 1, where 1 means identical and -1 means opposite. \n
    If both images belong to the same person, we call that pair of images as 'Genuine'.  \n
    'Genuine' refers to images of the same individual.  \n
    If both images belong to the different persons, we call that pair of images as 'Imposter'.  \n
    'Imposter' denotes images of different individuals. \n

    This method work by comparing the face images by using 9 face landmarks.  \n
    These landmarks are compared based on the  mean value and the ratio of 1s value.  \n
    The mean is the average value of the feature across all face images in the dataset, and the ratio of 1s is the proportion of face images that have the same value of the feature as the reference face image. \n 
    The lower the mean and higher the ratio of 1s, the more important the feature is. \n
    These total 9 landmarks are chin, lips, nose, eyes (left and right), eyebrows (left and right), cheeks (left and right).  \n
    A landmark is a point on the face that corresponds to a specific facial part, such as the nose, eye, lips etc.  \n
    Each landmark is composed of several attributes, such as the shape, size, color, and position of that landmark.  \n
    The most important landmarks for this comparison is {imp_landmarks}.  \n
    The least important landmarks for this comparison is {least_imp_landmarks}.  \n
    The importance of a landmark means that these landmarks of both face images are very similar, which supports our decision, while the least important means these landmarks are slightly different, which does not affect our decision. \n
    This importance score is calculated by analyzing the majority color of a that landmark and checking against the divided color range of COLORMAP_JET (https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html). \n


    Available explainable AI approaches are {avl_approaches}. \n
    {selected_approaches} technique(s) are used for this comparison. \n

    To interpret output images, these are the heatmaps generated by Explainable AI approaches. \n
    A heatmap is the graphical representation of data where values are dipicted by different colors. \n
    The colors on the heatmaps ranges from red to blue where each color represents the importance of the underlying landmark. \n
    The colors similar to red shows the most important regions of the face.
    The colors similar to blue shows least important regions of the face. \n
    Explainable Artificial Intelligence (XAI) is an approach in the development of AI systems that focuses on making the decision-making process of AI models understandable and transparent to humans. \n
    The Explainable Artificial Intelligence (XAI) used in this process are Minus Single, Minus Greedy, Plus Single, Plus Greedy, and Average Saliency approach. \n
    Minus Single is a XAI approach proposed in this paper (https://rb.gy/w84le3) which finds the most important regions of the face by removing a region first and then calculating its effect on similarity. \n
    Average Saliency approach is the best approach because it produces most stable results. \n

    To interpret output images, these are the heatmaps generated by Explainable AI approaches. \n
    A heatmap is the graphical representation of data where values are dipicted by different colors. \n
    The colors on the heatmaps ranges from red to blue where each color represents the importance of the underlying landmark. \n
    The colors similar to red shows the most important regions of the face.
    The colors similar to blue shows least important regions of the face. \n
    Explainable Artificial Intelligence (XAI) is an approach in the development of AI systems that focuses on making the decision-making process of AI models understandable and transparent to humans. \n
    The Explainable Artificial Intelligence (XAI) used in this process are Minus Single, Minus Greedy, Plus Single, Plus Greedy, and Average Saliency approach. \n
    Minus Single is a XAI approach proposed in this paper (https://rb.gy/w84le3) which finds the most important regions of the face by removing a region first and then calculating its effect on similarity. \n
    Average Saliency approach is the best approach because it produces most stable results. \n

    {get_landmark_importance('chin', df)}  \n
    {get_landmark_importance('lips', df)}  \n
    {get_landmark_importance('nose', df)}  \n
    {get_landmark_importance('left_eye', df)}  \n
    {get_landmark_importance('right_eye', df)}  \n
    {get_landmark_importance('left_eyebrow', df)}  \n
    {get_landmark_importance('right_eyebrow', df)}  \n
    {get_landmark_importance('left_cheek', df)}  \n
    {get_landmark_importance('right_cheek', df)}  \n

    """

    return full_context


def format_float(value):
    # Define a regex pattern that matches a float number
    # string value with spaces after the dot e.g. "1. 0"
    pattern=r'\d+\s*\.\s*\d+'

    # Use re.search() to find the first match of the pattern in the string
    match = re.search(pattern, value)

    # If a match is found, use float() to convert it to a float value
    if match:
        value = float(match.group().replace(' ', ''))
        return value
    else:
        return value


def ask_model_full_context(query, context, qa_tokenizer, qa_model):

    inputs = qa_tokenizer(query, context, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True,max_length=512, )
    with torch.no_grad():
        outputs = qa_model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

    answer = qa_tokenizer.decode(predict_answer_tokens)
    start_prob = softmax(outputs.start_logits, dim=-1)[0, answer_start_index].item()
    end_prob = softmax(outputs.end_logits, dim=-1)[0, answer_end_index].item()
    score = start_prob * end_prob
    
    return answer, score


def get_sub_context(query, detailed_text):

    # Convert the dataframe to text
    # detailed_text = dataframe_to_text(df, binary_df)

    # split paragraphs on newline character
    sentences=detailed_text.split('\n')

    # remove empty sentences
    sentences=[s for s in sentences if s!='']

    # embeddings of sentences
    index_em=get_embeddings(sentences)

    # embedding of query
    query_em=get_embeddings([query])[0] 

    # cosine similarity between query and all paragraphs (each para is related to individual landmark)
    sims=cosine_similarity([query_em], index_em)[0]

    # get the index of max value of sims array
    max_index = np.argmax(sims)
    sub_context = sentences[max_index]

    return sub_context


def ask_model(df, binary_df, query, detailed_text=None):

    if not detailed_text:
        # Convert the dataframe to text
        detailed_text = dataframe_to_text(df, binary_df)


    # split paragraphs on newline character
    sentences=detailed_text.split('\n')

    # remove empty sentences
    sentences=[s for s in sentences if s!='']

    context = ' '.join(sentences)

    answer, answer_conf=ask_model_full_context(query, context, qa_tokenizer, qa_model)
    if answer_conf < 0.3:
        # if the answer conf is low, get the specific para as context
        sub_context=get_sub_context(query, detailed_text)
        answer, answer_conf=ask_model_full_context(query, sub_context, qa_tokenizer, qa_model)

    # Step 6: format the float values which are like "2. 0", "1. 5" etc
    answer = format_float(answer)

    if answer: 
        return answer, context
    else: 
        answer='sorry, I cannot find the answer'
        return answer, context

