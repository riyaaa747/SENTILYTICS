
import streamlit as st
import matplotlib.pyplot as plt
from predict import predict_sentiment  # This function must return (sentiment, probs)
from transformers import BertTokenizer
from models.model import build_transformer
from config import *
import torch

# Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_transformer(
    SRC_VOCAB_SIZE,
    3,  # 3-class classification
    SRC_SEQ_LEN,
    TGT_SEQ_LEN,
    D_MODEL,
    N_LAYERS,
    H,
    DROPOUT,
    D_FF
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

# Set page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
          
            background-color: #1e293b;
        }
        .stApp {
            background: linear-gradient(to right, #1e293b, #334155);
            color: white;
        }
        .stTextInput > div > div > input {
            background-color: #334155;
            color: white;
        }
        .stButton>button {
            background-color: #6366f1;
            color: white;
            font-weight: bold;
            border-radius: 0.5rem;
            transition: 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #4f46e5;
            transform: scale(1.03);
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("## 🧠 Sentiment Analyzer")
st.markdown("### Analyze the sentiment of any sentence instantly!")
st.markdown("Let the AI read the mood of your message 🎯")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# User input
text_input = st.text_input("Type your sentence here 👇")

# Predict button
if st.button("🔍 Analyze Sentiment"):
    if not text_input.strip():
        st.warning("Please enter a sentence 💬")
    else:
        with st.spinner("Analyzing... 🤖"):
            sentiment, probs = predict_sentiment(text_input, model, tokenizer, device)

            emoji = "😐"
            color = "yellow"
            if sentiment == "Positive":
                emoji = "🥳"
                color = "green"
            elif sentiment == "Negative":
                emoji = "💔"
                color = "red"
            elif sentiment == "Neutral":
                emoji = "😐"
                color = "orange"

        # Final sentiment result
        st.success(f"*Sentiment:* :{color}[{sentiment}] {emoji}")

        # Show confidence scores
        st.markdown("### 🔢 Sentiment Confidence Scores")
        labels = ["Negative", "Positive", "Neutral"]
        for i in range(3):
            st.markdown(f"**{labels[i]}**: `{probs[i].item() * 100:.2f}%`")

        
        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie([p.item() * 100 for p in probs], labels=labels, autopct='%1.1f%%', startangle=90, colors=["red", "green", "yellow"])
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

        # Display pie chart
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("<center>Made with ❤ using Streamlit</center>", unsafe_allow_html=True)


