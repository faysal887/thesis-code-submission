
# https://github.com/streamlit/llm-examples/blob/main/Chatbot.py
import re
import pandas as pd
import streamlit as st

# from thesis.application.table_question_answer import ask_model as table_ask_model
from utils.question_answer.paragraph_question_answer import ask_model as para_ask_model
from utils.question_answer.paragraph_question_answer import dataframe_to_text

try:
    TABLE=st.session_state['final_display_df']
    BINARY_DF=st.session_state['binary_decision_df']
except:
    TABLE=pd.DataFrame()
    BINARY_DF=pd.DataFrame()
    
common_questions = {
    "Most Important Landmark": "What is the most important Landmark?",
    "Least Important Landmark": "What is the least important Landmark?",
    # "Importance of Nose": "What is the importance of nose?",
    # "Importance of Lips": "What is the importance of lips?",
    # "Mean Calculation": "how is the mean score calculated?",
    # "Criteria for Importance":"What is the criteria for importance calculation?",
    # "Metric for Importance":"What are the metrics of importance?",
    # "Importance Calculation":"How importance is calculated for a landmark?",
    # Add more questions as needed
}

# Function to handle question selection
def handle_question_selection():
    if selected_question_key:
        selected_question = common_questions[selected_question_key]
        st.session_state.messages.append({"role": "user", "content": selected_question})
        generate_response(selected_question)

# Generate a response for a given message
def generate_response(faq_query):
    # Simulated bot response (replace with your response logic)
    # faq_answer = table_ask_model(st.session_state.table, faq_query)
    faq_answer, context = para_ask_model(TABLE, BINARY_DF, faq_query)
    # response = f"Response to FAQ: {faq_answer} \n[{context}]"
    response = f"{faq_answer}"
    st.session_state.messages.append({"role": "assistant", "content": response})

def reset_conversation():
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


def show_detailed_text():
    text=dataframe_to_text(TABLE, BINARY_DF)
    st.session_state.messages.append({"role": "assistant", "content": text})

def show_detailed_text_stripped():
    text=dataframe_to_text(TABLE, BINARY_DF)
    st.session_state.messages.append({"role": "assistant", "content": text.split('\n')})

with st.sidebar:
    st.markdown("*******")

    st.title("🤔 FAQs")
    selected_question_key = st.selectbox("Common Questions", list(common_questions.keys()))

    # Button to submit the selected question
    st.sidebar.button("🔍 Ask Selected Question", on_click=handle_question_selection)
    st.sidebar.button("📖 Show Detailed Text", on_click=show_detailed_text)
    st.sidebar.button(":bookmark_tabs: Show Detailed Text Stripped", on_click=show_detailed_text_stripped)
    st.button('🔄 reset', on_click=reset_conversation)

st.title("💬 Chatbot")
st.caption("🚀 Chatbot to discuss XAI results")

# from PIL import Image
# Sample images for illustration
# image1 = Image.open("/data/faysal/thesis/application/output/output_images/merged_plus_merkel_K_simon_biles_A.png")
# image2 = Image.open("/data/faysal/thesis/application/output/output_images/contour_face_average_brad_brad_glasses.png")

# Display two images in a single row
# st.image([image1], width=700, caption=['Plus Single Saliency Output'])

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    ###################### GET RESPONSE FROM QA MODEL ######################
    # msg='hello this is a response'


    # answer=table_ask_model(table, query)
    answer, context=para_ask_model(TABLE, BINARY_DF, query)
    # answer='Hello world, this is my response'
    ########################################################################

    st.session_state.messages.append({"role": "assistant", "content": answer})
    # st.chat_message("assistant").write(f"{answer}\n[{context}]")
    st.chat_message("assistant").write(f"{answer}")

