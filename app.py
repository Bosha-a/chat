# import streamlit as st
# import numpy as np 
# from sklearn.metrics.pairwise import cosine_similarity
# import pyarabic
# import pyarabic.araby as araby
# import re 
# import pandas as pd 
# from sentence_transformers import SentenceTransformer
# import torch
# import os


# # Load the dataset
# df = pd.read_csv(r'fatwaa_2.csv')

# # Check if CUDA is available
# cuda_available = torch.cuda.is_available()
# print(f"CUDA Available: {cuda_available}")

# if cuda_available:
#     # Get the current CUDA device
#     cuda_device = torch.cuda.current_device()
#     print(f"Current CUDA Device: {torch.cuda.get_device_name(cuda_device)}")
#     print(f"Device Count: {torch.cuda.device_count()}")
    
#     # Set the device
#     device = torch.device("cuda")
# else:
#     print("CUDA not available. Using CPU instead.")
#     device = torch.device("cpu")

# def load_arbert():
#     print("Loading ARBERT model with device:", device)
#     model = SentenceTransformer('UBC-NLP/ARBERT', device=device)
#     print(f"Model loaded successfully on {device}")
#     return model

# # Choose which model to use
# model = load_arbert()

# # Processing text 
# def preprocess_text(text):
#     """
#     Comprehensive Arabic text preprocessing function
#     """
#     # 1. Remove URLs, emails, hashtags, and mentions
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = re.sub(r'\S+@\S+', '', text)
#     text = re.sub(r'#\S+', '', text)
#     text = re.sub(r'@\S+', '', text)
    
#     # 2. Remove non-Arabic characters except spaces
#     text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    
#     # 4. Remove diacritics (tashkeel)
#     text = araby.strip_tashkeel(text)
    
#     # 5. Remove tatweel (elongation character)
#     text = araby.strip_tatweel(text)
    
#     # 6. Remove consecutive spaces
#     text = re.sub(r'\s+', ' ', text)
    
#     # 7. Trim leading and trailing spaces
#     text = text.strip()
    
#     return text

# # Preprocess all questions in the dataset
# print("Preprocessing questions...")
# df['processed_question'] = df['question'].apply(preprocess_text)

# # Generate embeddings for all questions
# print("Generating embeddings for all questions (this may take a while)...")
# with torch.no_grad():  # Disable gradient calculation for efficiency
#     question_embeddings = model.encode(df['processed_question'].tolist())
# print(f"Generated embeddings with shape: {question_embeddings.shape}")

# def get_response(user_query, threshold=0.60):
#     """
#     Find the most similar question and return its answer
#     threshold: minimum similarity score to consider a match
#     """
#     # Preprocess the user query
#     processed_query = preprocess_text(user_query)
    
#     # Generate embedding for the user query
#     with torch.no_grad():
#         query_embedding = model.encode([processed_query])[0]
    
#     # Calculate similarity between query and all questions
#     # THIS IS THE FIXED LINE - compare with question_embeddings, not with itself
#     similarities = cosine_similarity([query_embedding], question_embeddings)[0]
    
#     # Find the most similar question
#     max_similarity_idx = np.argmax(similarities)
#     max_similarity = similarities[max_similarity_idx]
    
#     # Check if the similarity exceeds our threshold
#     if max_similarity >= threshold:
#         return df['answer'][max_similarity_idx], max_similarity, df['question'][max_similarity_idx]
#     else:
#         return "اسف لا املك معلومات كافية, نظرا لعدم الافتاء في الدين", max_similarity, ""

# # def chatbot():
# #     """Simple chatbot interface"""
# #     print("بوت الفتاوى: مرحبا! كيف يمكنني مساعدتك اليوم؟ (اكتب 'خروج' للمغادرة)")
    
# #     while True:
# #         user_input = input("أنت: ")
        
# #         if user_input.lower() in ['خروج', 'exit']:
# #             print("بوت الفتاوى: شكرا للدردشة معنا. مع السلامة!")
# #             break
            
# #         response, confidence, matched_question = get_response(user_input)
# #         print(f"بوت الفتاوى: {response} [نسبة الثقة: {confidence:.2f}]")
# #         if matched_question:
# #             print(f"السؤال المطابق: {matched_question}")
# #         print("-" * 50)

# # # Run the chatbot
# # print("Initializing chatbot...")
# # chatbot()



# # Streamlit chat interface
# st.set_page_config(page_title="بوت الفتاوى", layout="centered")

# st.title("Fatwaa Chatbot - بوت الفتاوى")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # User input
# user_input = st.text_input("✍️ اكتب سؤالك هنا:", key="user_input")

# if user_input:
#     # Get response
#     response, confidence, matched_question = get_response(user_input)

#     # Save user and bot messages
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     st.session_state.messages.append({"role": "bot", "content": f"{response}"})

# # Display chat history
# for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         st.markdown(f"**أنت:** {msg['content']}")
#     else:
#         st.markdown(f"**بوت الفتاوى:** {msg['content']}")

# # Footer note
# st.markdown("---")
# st.markdown("صنع ب ❤️ بواسطة عبدالله بشاري")
# st.markdown("اللهم اجعله خالص لوجه الله")


# ----------

import streamlit as st
import numpy as np
import pandas as pd
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import pyarabic.araby as araby

# Load dataset
df = pd.read_csv("fatwaa_2.csv")

# Device setup (Streamlit Cloud is CPU-only)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner="🚀 Loading ARBERT model...")
def load_arbert():
    return SentenceTransformer('UBC-NLP/ARBERT', device=device)

model = load_arbert()

# Arabic preprocessing
def preprocess_text(text):
    text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+|#\S+|@\S+', '', text)  # Remove links/emails/hashtags
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic characters and spaces
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Preprocess questions
df['processed_question'] = df['question'].apply(preprocess_text)

# Generate question embeddings (cache to avoid recomputation)
@st.cache_resource(show_spinner="🔎 Generating embeddings...")
def generate_embeddings():
    with torch.no_grad():
        return model.encode(df['processed_question'].tolist())

question_embeddings = generate_embeddings()

# Similarity function
def get_response(user_query, threshold=0.60):
    processed_query = preprocess_text(user_query)
    with torch.no_grad():
        query_embedding = model.encode([processed_query])[0]
    similarities = cosine_similarity([query_embedding], question_embeddings)[0]
    idx = np.argmax(similarities)
    score = similarities[idx]
    if score >= threshold:
        return df['answer'][idx], score, df['question'][idx]
    return "اسف لا املك معلومات كافية, نظرا لعدم الافتاء في الدين", score, ""

# Streamlit interface
st.set_page_config(page_title="بوت الفتاوى", layout="centered")
st.title("Fatwaa Chatbot - بوت الفتاوى")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("✍️ اكتب سؤالك هنا:", key="user_input")

if user_input:
    response, confidence, matched = get_response(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "bot", "content": response})

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**أنت:** {msg['content']}")
    else:
        st.markdown(f"**بوت الفتاوى:** {msg['content']}")

st.markdown("---")
st.markdown("صنع ب ❤️ بواسطة عبدالله بشاري")
st.markdown("اللهم اجعله خالص لوجه الله")
