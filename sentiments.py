# -*- coding: utf-8 -*-
"""sentiments.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mVERhCqkDaoFgppBEnlkabimbj2iIMnI

# DATASET

##  Dataset 1: Movie Reviews dataset
"""

file_path = '/content/drive/MyDrive/ds/movie_review.csv'

# Load the dataset
data = pd.read_csv(file_path)

# Display the first few rows to confirm it's loaded correctly
data.head()

data = data[['text', 'tag']]
data.head()

# Using .shape to get the number of rows
num_samples = data.shape[0]
print("Number of data samples:", num_samples)

# Count the number of samples in each category
category_counts = data['tag'].value_counts()

# Display the counts
print(category_counts)

"""## Dataset 2: Twitter Tweets"""

# Load the dataset with specified column names
twitter_file_path = "/content/drive/MyDrive/ds/twitter_training.csv"
twitter_data = pd.read_csv(twitter_file_path)


# Display the first few rows to verify the changes
twitter_data.head()

twitter_data.columns

twitter_data = twitter_data[['Positive', 'im getting on borderlands and i will murder you all ,' ]]
twitter_data.head()

twitter_data.rename(columns={'Positive': 'tag'}, inplace=True)
twitter_data.rename(columns={'im getting on borderlands and i will murder you all ,': 'text'}, inplace=True)
twitter_data.head()

# Using .shape to get the number of rows
num_samples = twitter_data.shape[0]
print("Number of data samples:", num_samples)

category_counts = twitter_data['tag'].value_counts()

# Display the counts
print(category_counts)



"""## Dataset 3: Amazon Product Reviews"""

# Load the dataset with specified column names
amazon_file_path = "/content/drive/MyDrive/amazon.csv"
amazon_data = pd.read_csv(amazon_file_path)


# Display the first few rows to verify the changes
amazon_data.head()

amazon_data.rename(columns={'class_index': 'tag'}, inplace=True)
amazon_data.rename(columns={'review_text': 'text'}, inplace=True)
amazon_data.head()

amazon_data = amazon_data[["text","tag"]]
amazon_data.head()

# Using .shape to get the number of rows
num_samples = amazon_data.shape[0]
print("Number of data samples:", num_samples)

category_counts = amazon_data['tag'].value_counts()

# Display the counts
print(category_counts)



"""# Importing Libraries"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from bs4 import BeautifulSoup
import os
from groq import Groq
from sklearn.feature_extraction import text
from collections import Counter
import torch
from googleapiclient.discovery import build
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

"""# Merging Datasets"""

data.head()

data["tag"].unique()

data['tag'].replace({'pos': 0, 'neg': 1}, inplace=True)
data.head()

data['tag'].value_counts()

twitter_data.head()

twitter_data["tag"].unique()

twitter_data['tag'].replace({'Positive': 0, 'Negative': 1, 'Neutral': 2}, inplace=True)

# Drop rows where 'tag' is 'Irrelevant'
twitter_data = twitter_data[twitter_data['tag'] != 'Irrelevant']
twitter_data["tag"].unique()

twitter_data["tag"].unique()
array([0, 2, 1], dtype=object)

amazon_data.head()

amazon_data["tag"].unique()

amazon_data['tag'].replace({2: 0}, inplace=True)

amazon_data['tag'].unique()

# Sample 5000 rows randomly for each tag
amazon_data_0 = amazon_data[amazon_data['tag'] == 0].sample(n=5000, random_state=42)
amazon_data_1 = amazon_data[amazon_data['tag'] == 1].sample(n=5000, random_state=42)

# Concatenate the two samples to form a new DataFrame
amazon_data_new = pd.concat([amazon_data_0, amazon_data_1])

# Shuffle the rows in the new DataFrame to mix the tags
amazon_data_new = amazon_data_new.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the structure of the new DataFrame to confirm the changes
print(amazon_data_new.shape)
print(amazon_data_new['tag'].value_counts())
amazon_data_new.head()

# Sample 5000 rows randomly for each of tags 0 and 1
twitter_data_0 = twitter_data[twitter_data['tag'] == 0].sample(n=5000, random_state=42)
twitter_data_1 = twitter_data[twitter_data['tag'] == 1].sample(n=5000, random_state=42)

# Select all rows with tag 2
twitter_data_2 = twitter_data[twitter_data['tag'] == 2]

# Concatenate the samples to form a new DataFrame
twitter_data_new = pd.concat([twitter_data_0, twitter_data_1, twitter_data_2])

# Shuffle the rows in the new DataFrame to mix the tags
twitter_data_new = twitter_data_new.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the structure of the new DataFrame to confirm the changes
print(twitter_data_new.shape)
twitter_data_new['tag'].value_counts()

twitter_data_new

# Sample 5000 rows randomly for each of tags 0 and 1
amazon_data_0 = amazon_data[amazon_data['tag'] == 0].sample(n=5000, random_state=42)
amazon_data_1 = amazon_data[amazon_data['tag'] == 1].sample(n=5000, random_state=42)

# Concatenate the samples to form a new DataFrame
amazon_data_new = pd.concat([amazon_data_0, amazon_data_1])

# Shuffle the rows in the new DataFrame to mix the tags
amazon_data_new = amazon_data_new.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the structure of the new DataFrame to confirm the changes
print(amazon_data_new.shape)
print(amazon_data_new['tag'].value_counts())
amazon_data_new

# Sample 5000 rows randomly for each of tags 0 and 1
movie_data_0 = data[data['tag'] == 0].sample(n=5000, random_state=42)
movie_data_1 = data[data['tag'] == 1].sample(n=5000, random_state=42)

# Concatenate the samples to form a new DataFrame
movie_data_new = pd.concat([movie_data_0, movie_data_1])

# Shuffle the rows in the new DataFrame to mix the tags
movie_data_new = movie_data_new.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the structure of the new DataFrame to confirm the changes
print(movie_data_new.shape)
print(movie_data_new['tag'].value_counts())
movie_data_new.head()

combined_data = pd.concat([movie_data_new, amazon_data_new, twitter_data_new])

# Shuffle the rows in the combined DataFrame to randomize the order
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the structure of the combined DataFrame to confirm the changes
print(combined_data.shape)
combined_data.head()

combined_data["tag"].unique()

combined_data["tag"].value_counts()

# Downsample each category to 10,000 samples
tag_0_downsampled = combined_data[combined_data['tag'] == 0].sample(n=10000, random_state=42)
tag_1_downsampled = combined_data[combined_data['tag'] == 1].sample(n=10000, random_state=42)
tag_2_downsampled = combined_data[combined_data['tag'] == 2].sample(n=10000, random_state=42)

# Combine the downsampled DataFrames
balanced_data = pd.concat([tag_0_downsampled, tag_1_downsampled, tag_2_downsampled])

# Shuffle the rows in the balanced DataFrame to randomize the order
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the structure of the balanced DataFrame to confirm the changes
print(balanced_data.shape)
print(balanced_data['tag'].value_counts())

# Save the balanced DataFrame to a CSV file
balanced_data.to_csv('/content/drive/MyDrive/balanced_data.csv', index=False)

# Print a confirmation message
print("The balanced data has been successfully saved to a CSV file.")



"""# DATA PREPROCESSING AND EXPLORATORY DATA ANALYSIS"""

balanced_data = pd.read_csv('/content/drive/MyDrive/balanced_data.csv')
balanced_data.head()

"""## Removing Null and Duplicates"""

print("Initial number of rows:", balanced_data.shape[0])

# 1. Removing Null Values
balanced_data.dropna(subset=['text', 'tag'], inplace=True)
print("Number of rows after removing nulls:", balanced_data.shape[0])

# 2. Removing Duplicate Rows
balanced_data.drop_duplicates(subset=['text', 'tag'], inplace=True)
print("Number of rows after removing duplicates:", balanced_data.shape[0])

balanced_data["tag"].value_counts()

balanced_data.info()

import seaborn as sns
import matplotlib.pyplot as plt

# Visualizing the distribution of tags
plt.figure(figsize=(8, 5))
sns.countplot(x='tag', data=balanced_data)
plt.title('Distribution of Tags')
plt.xlabel('Tag')
plt.ylabel('Frequency')
plt.show()

"""## Insights about Comments lenght"""

# Adding a new column for text length
balanced_data['text_length'] = balanced_data['text'].apply(len)

# Plotting the distribution of text lengths
plt.figure(figsize=(10, 6))
sns.histplot(balanced_data['text_length'], bins=30, kde=True)
plt.title('Distribution of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

"""## Word Cloud Analysis"""

from wordcloud import WordCloud

# Function to generate word cloud for each tag
def generate_word_cloud(tag):
    text = " ".join(review for review in balanced_data[balanced_data['tag'] == tag]['text'])
    wordcloud = WordCloud(background_color='white').generate(text)
    plt.figure(figsize=(8, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Tag {tag}')
    plt.show()

# Generate word clouds for each tag
for tag in balanced_data['tag'].unique():
    generate_word_cloud(tag)

"""## TF-Idf Analysis"""

from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_top_words(tag):
    text_data = balanced_data[balanced_data['tag'] == tag]['text']
    vectorizer = TfidfVectorizer(stop_words='english', max_features=30)
    tfidf_matrix = vectorizer.fit_transform(text_data)
    feature_names = vectorizer.get_feature_names_out()
    return feature_names

# Display top TF-IDF features for each tag
for tag in balanced_data['tag'].unique():
    print(f'Top TF-IDF words for tag {tag}: {tfidf_top_words(tag)}')

"""## HTML-Tags, Stop Words Removal and Lowering of TEXT"""

def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove special characters and numbers (optional, customize as needed)
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)

    # Convert to lowercase
    text = text.lower()
    return text

balanced_data['text'] = balanced_data['text'].apply(clean_text)

"""## BERT Tokenization"""

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

max_length = 512

tokenized_data = tokenizer(
    list(balanced_data['text']),       # Input text
    padding='max_length',              # Pad sequences to `max_length`
    truncation=True,                   # Truncate sequences longer than `max_length`
    max_length=max_length,             # Maximum sequence length
    return_tensors='pt'                # Return PyTorch tensors
)

# Extracting input_ids and attention masks from the tokenized data
input_ids = tokenized_data['input_ids']
attention_masks = tokenized_data['attention_mask']

"""## Train Test Splitting"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Splitting the data into train and validation sets
labels = torch.tensor(balanced_data['tag'].to_numpy())
input_ids = torch.tensor(input_ids)  # Make sure this is already a tensor or convert it
attention_masks = torch.tensor(attention_masks)  # Make sure this is already a tensor or convert it

# Create train-validation split
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels, random_state=2018, test_size=0.1)

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

# Create the DataLoader for our validation set
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=32)

# Load DistilBERT for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(balanced_data['tag'].unique()))
#model.cuda()  # Uncomment this line if using a GPU

# Set up the optimizer and the learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Function to calculate the accuracy of predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

"""# MODEL TRAINING AND TESTING"""

# Training loop
model.train()
for epoch_i in range(0, epochs):
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)
        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss}")

# Validation of the model after training
model.eval()
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0

for batch in validation_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_input_mask)
    logits = outputs.logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

print(f"Validation Accuracy: {eval_accuracy/nb_eval_steps}")

# Calculate precision, recall, and F1-score
predictions = []
true_labels = []
for batch in validation_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_input_mask)
    logits = outputs.logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(logits)
    true_labels.append(label_ids)

# Flatten the predictions and true values for aggregate metrics
flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = np.concatenate(true_labels, axis=0)

# Calculate and display precision, recall, and F1-score
precision, recall, f1, _ = precision_recall_fscore_support(flat_true_labels, flat_predictions, average='weighted')
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

f1

import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your model (assuming it's already loaded and set to evaluation mode with model.eval())

# Assuming 'input_ids', 'attention_masks', and 'labels' are your full training datasets already loaded as tensors
dataset = TensorDataset(input_ids, attention_masks, labels)
sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

correct = 0
total = 0

with torch.no_grad():
    for batch in dataloader:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

        outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        total += b_labels.size(0)
        correct += (predictions == b_labels).sum().item()

accuracy = correct / total
print(f'Training Accuracy: {accuracy:.4f}')

import matplotlib.pyplot as plt

# Accuracy values
training_accuracy = 0.8300  # The training accuracy you provided
validation_accuracy = 0.765  # A hypothetical validation accuracy you might have

# Labels for the bars
labels = ['Training', 'Validation']

# Heights of the bars
heights = [training_accuracy, validation_accuracy]

# Creating the bar graph
plt.figure(figsize=(8, 5))  # Set the figure size
plt.bar(labels, heights, color=['blue', 'green'])  # Set different colors for each bar
plt.xlabel('Type of Accuracy')  # Label on the x-axis
plt.ylabel('Accuracy')  # Label on the y-axis
plt.title('Comparison of Training and Validation Accuracies')  # Title of the plot
plt.ylim(0.5, 0.85)  # Set y-axis limits to make differences more noticeable
for i, v in enumerate(heights):
    plt.text(i, v + 0.005, f"{v:.4f}", ha='center', va='bottom')  # Adding text labels on bars
plt.show()

import matplotlib.pyplot as plt

# Data for plotting
epochs = [1, 2, 3]
training_loss = [0.6921257237979592, 0.4826446905424354, 0.3545267965151212]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, training_loss, marker='o', linestyle='-', color='r')
plt.title('Training Loss Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.grid(True)
plt.xticks(epochs)  # Ensure all epochs are marked
plt.show()

import matplotlib.pyplot as plt

# Metrics
metrics = ['Precision', 'Recall', 'F1-score']
values = [0.7639453006932819, 0.7647863247863248, 0.7640698533571587]

# Create a bar plot
plt.figure(figsize=(8, 5))  # Set the figure size
plt.bar(metrics, values, color=['blue', 'green', 'red'])  # Set different colors for each bar
plt.xlabel('Metrics')  # Label on the x-axis
plt.ylabel('Values')  # Label on the y-axis
plt.title('Model Performance Metrics')  # Title of the plot
plt.ylim(0.75, 0.77)  # Limit y-axis to make differences more noticeable
for i, v in enumerate(values):
    plt.text(i, v + 0.0005, f"{v:.4f}", ha='center', va='bottom')  # Adding text labels on bars
plt.show()

model_path = "/content/drive/MyDrive/entire_model.pth"
torch.save(model, model_path)

print("Model has been saved to", model_path)

"""## Model Loading and Manual Evaluation"""

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model architecture
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)  # Adjust num_labels accordingly
model.to(device)

# Load the saved weights
model_path = '/content/drive/MyDrive/model_Epoch_2.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

"""## Manual Testing on Negative comment"""

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Testing on Negative Comment
text = "Not that good"

# Encode the text using the same tokenizer used during training
encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

# Convert logits to probabilities (optional)
probabilities = torch.nn.functional.softmax(logits, dim=1)
predicted_label = torch.argmax(probabilities, dim=1)
if predicted_label.item() == 1:
  print("Predicted label: Negative")

"""## Manual Testing on Positive Comment"""

# Testing on Positive Comment
text = "This is very good lecture"

# Encode the text using the same tokenizer used during training
encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

# Convert logits to probabilities (optional)
probabilities = torch.nn.functional.softmax(logits, dim=1)
predicted_label = torch.argmax(probabilities, dim=1)
if predicted_label.item() == 0:
  print("Predicted label: Positive")

"""## Manual Testing on Neutral comment"""

# Testing on Positive Comment
text = "okok"

# Encode the text using the same tokenizer used during training
encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

# Convert logits to probabilities (optional)
probabilities = torch.nn.functional.softmax(logits, dim=1)
predicted_label = torch.argmax(probabilities, dim=1)
if predicted_label.item() == 2:
  print("Predicted label: Normal")

"""# E-Learning Pipeline"""

pip install --upgrade google-api-python-client

from googleapiclient.discovery import build
import re

api_key = 'AIzaSyBd-5Ufc-3I0lrqfcJuZYBQdxs_5UeYgbg'

# Initialize the YouTube client
youtube = build('youtube', 'v3', developerKey=api_key)

def extract_video_id(url):
    # This regular expression will match any possible YouTube URL and extract the video ID
    regex = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\s*[^\/\n\s]+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(regex, url)
    return match.group(1) if match else None

def get_video_comments(video_id, max_comments=50):
    comments = []
    results = youtube.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText').execute()
    count = 0  # Counter to track the number of comments

    while results and count < max_comments:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            count += 1
            if count >= max_comments:
                break  # Break from the for loop if max_comments reached

        if 'nextPageToken' in results and count < max_comments:
            results = youtube.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText', pageToken=results['nextPageToken']).execute()
        else:
            break  # Break from the while loop if max_comments reached or no more pages

    return comments

# Provide the full YouTube video URL here
video_url = 'https://www.youtube.com/watch?v=dz7Ntp7KQGA'
video_id = extract_video_id(video_url)  # Extract the video ID from the URL

if video_id:
    comments = get_video_comments(video_id, max_comments=50)

    # Output comments
    for comment in comments:
        print(comment)
else:
    print("Invalid YouTube URL")

len(comments)

comments

import re

def remove_emojis(text):
    # Unicode ranges for emojis and some symbols that encompass most emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess_and_predict(comments):
    # Remove emojis and encode the comments
    cleaned_comments = [remove_emojis(comment) for comment in comments]
    encoded_comments = tokenizer(cleaned_comments, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Move tensors to the CPU
    input_ids = encoded_comments['input_ids'].to(device)
    attention_mask = encoded_comments['attention_mask'].to(device)

    # Predict using the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)

    return predictions.cpu().numpy()



# Get predictions for these comments
predictions = preprocess_and_predict(comments)

from collections import Counter

# Count predictions
prediction_counts = Counter(predictions)
print("Prediction Summary:")
print(f"Positive Comments: {prediction_counts.get(0, 0)}")
print(f"Negative Comments: {prediction_counts.get(1, 0)}")
print(f"Neutral Comments: {prediction_counts.get(2, 0)}")

# Example of getting dynamic values:
positive_count = prediction_counts.get(0, 0)
negative_count = prediction_counts.get(1, 0)
neutral_count = prediction_counts.get(2, 0)

# Labels and dynamically fetched counts
labels = ['Positive Comments', 'Negative Comments', 'Neutral Comments']
counts = [positive_count, negative_count, neutral_count]

# Creating the bar graph
plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=['green', 'red', 'blue'])

# Adding titles and labels
plt.title('Comment Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')

# Display the counts on top of the bars
for i, count in enumerate(counts):
    plt.text(i, count + 1, str(count), ha='center', va='bottom')

# Display the bar graph
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

negative_comments = [comment for comment, prediction in zip(comments, predictions) if prediction == 1]
print("Negative comments:")
for comment in negative_comments:
    print(comment)

positive_comments = [comment for comment, prediction in zip(comments, predictions) if prediction == 0]
print("Postive comments:")
for comment in positive_comments:
    print(comment)

neutral_comments = [comment for comment, prediction in zip(comments, predictions) if prediction == 2]
print("Neutral comments:")
for comment in neutral_comments:
    print(comment)

import os
from groq import Groq

# Initialize Groq client with your API key
client = Groq(
    api_key="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",
)


def get_insights(comments, category):
    # Construct the prompt to summarize insights from the comments
    prompt = f"Summarize the key insights from these {category} comments to help create suggestions:"
    # Add comments to the messages list
    messages = [{"role": "system", "content": prompt}]
    messages.extend([{"role": "user", "content": comment} for comment in comments])
    print(messages)
    # Create chat completion
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )
    print("i am here")

    # Extract the completion response
    return chat_completion.choices[0].message.content

# Assuming 'positive_comments', 'negative_comments', 'neutral_comments' are defined
positive_insights = get_insights(positive_comments, 'positive')
# negative_insights = get_insights(negative_comments, 'negative')
# neutral_insights = get_insights(neutral_comments, 'neutral')

print("Positive Insights:", positive_insights)
# print("Negative Insights:", negative_insights)
# print("Neutral Insights:", neutral_insights)

import os
from groq import Groq

# Initialize Groq client with your API key
client = Groq(
    api_key="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",
)

def get_insights(comments):
    # Construct the prompt to summarize insights from all comments
    prompt = " I am sending you comments for a video link. These are the negative labeld comments by the model. You have to analuyze these comments and summazrize all comments with most useful information so that it may be used later to create suggestions."

    # Concatenate all comments into a single string
    all_comments_text = " ".join(comments)  # Join all comments with a space

    # Prepare the messages list for the API call
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": all_comments_text}
    ]

    # Create chat completion
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )

    # Extract the completion response
    return chat_completion.choices[0].message.content

# Example usage
# Assuming 'comments' is a list of all comments
insights = get_insights(negative_comments)

print("Insights:", insights)

import os
from groq import Groq

# Initialize Groq client with your API key
client = Groq(
    api_key="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",
)

def get_insights(comments):
    # Construct the prompt to summarize insights from all comments
    prompt = " I am sending you comments for a video link. These are the poditively labeld comments by the model. You have to analuyze these comments and summazrize all comments with most useful information so that it may be used later to create suggestions."

    # Concatenate all comments into a single string
    all_comments_text = " ".join(comments)  # Join all comments with a space

    # Prepare the messages list for the API call
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": all_comments_text}
    ]

    # Create chat completion
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )

    # Extract the completion response
    return chat_completion.choices[0].message.content

# Example usage
# Assuming 'comments' is a list of all comments
insights = get_insights(positive_comments)

print("Insights:", insights)