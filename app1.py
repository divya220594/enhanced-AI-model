import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from googleapiclient.discovery import build
import re
import matplotlib.pyplot as plt
from collections import Counter
import os
from groq import Groq

# Set up the device
device = torch.device("cpu")

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Initialize the model architecture
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.to(device)

# Load the saved weights
model_path = 'model_Epoch_2.pth'
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Initialize the YouTube client
api_key = 'AIzaSyBd-5Ufc-3I0lrqfcJuZYBQdxs_5UeYgbg'
youtube = build('youtube', 'v3', developerKey=api_key)

# Streamlit App
st.title('YouTube Video Comment Sentiment Analysis')

def extract_video_id(url):
    regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\s*[^\/\n\s]+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(regex, url)
    return match.group(1) if match else None

def get_video_comments(video_id, max_comments=50):
    comments = []
    results = youtube.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText').execute()
    count = 0

    while results and count < max_comments:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            count += 1
            if count >= max_comments:
                break

        if 'nextPageToken' in results and count < max_comments:
            results = youtube.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText', pageToken=results['nextPageToken']).execute()
        else:
            break

    return comments

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F700-\U0001F77F"
                               u"\U0001F780-\U0001F7FF"
                               u"\U0001F800-\U0001F8FF"
                               u"\U0001F900-\U0001F9FF"
                               u"\U0001FA00-\U0001FA6F"
                               u"\U0001FA70-\U0001FAFF"
                               u"\U00002702-\U000027B0"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess_and_predict(comments):
    cleaned_comments = [remove_emojis(comment) for comment in comments]
    encoded_comments = tokenizer(cleaned_comments, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = encoded_comments['input_ids'].to(device)
    attention_mask = encoded_comments['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.cpu().numpy()

def plot_sentiment_distribution(predictions):
    prediction_counts = Counter(predictions)
    labels = ['Positive', 'Negative', 'Neutral']
    counts = [prediction_counts.get(0, 0), prediction_counts.get(1, 0), prediction_counts.get(2, 0)]
    fig, ax = plt.subplots()
    ax.bar(labels, counts, color=['green', 'red', 'blue'])
    ax.set_title('Comment Sentiment Distribution')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Comments')
    for i, count in enumerate(counts):
        ax.text(i, count + 1, str(count), ha='center', va='bottom')
    st.pyplot(fig)

# Initialize Groq client
groq_api_key = "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"
client = Groq(api_key=groq_api_key)

def get_insights(comments):
    prompt = "I am sending you comments for a video link. These are the positively labeled comments by the model. You have to analyze these comments and summarize all comments with most useful information so that it may be used later to create suggestions."
    all_comments_text = " ".join(comments)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": all_comments_text}
    ]
    chat_completion = client.chat.completions.create(messages=messages, model="llama3-8b-8192")
    return chat_completion.choices[0].message.content

# Main App
video_url = st.text_input('Enter YouTube Video URL', '')

if st.button('Analyze Comments'):
    video_id = extract_video_id(video_url)
    if video_id:
        st.session_state.comments = get_video_comments(video_id)
        st.session_state.predictions = preprocess_and_predict(st.session_state.comments)
        plot_sentiment_distribution(st.session_state.predictions)
        st.write("Here are some sample comments:")
        for comment in st.session_state.comments[:5]:
            st.text(comment)
    else:
        st.error("Invalid YouTube URL")

if st.button('Get Insights on Positive Comments') and st.session_state.comments is not None and st.session_state.predictions is not None:
    positive_comments = [comment for comment, pred in zip(st.session_state.comments, st.session_state.predictions) if pred == 0]
    insights = get_insights(positive_comments)
    st.write(insights)
