import pandas as pd
import os

from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score
from transformers import AdamW
from multi_model import MultiTaskBERT,preprocess_text
import nltk
        
def preprocess_data(texts, labels1, labels2, tokenizer, max_length=128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return encodings, labels1, labels2



label_encoder = LabelEncoder()
nltk.download('punkt_tab')
nltk.download('stopwords')
PATH="SAVED_MODEL_MULTICLASS.pt"
DATASET_PATH="dataset.xlsx"
data_df=pd.read_excel(DATASET_PATH,engine="openpyxl", sheet_name=None)
train_data_df=data_df["Train"]
test_data_df=data_df["Test"]
train_data_df.drop(6,inplace=True)
train_data_df['cleaned_tweet_text'] = train_data_df['tweet_text'].apply(preprocess_text)
train_data_df["emotion_in_tweet_is_directed_at"].fillna(value="Other Google product or service", inplace=True)
train_data_df['encoded_emotion_in_tweet_is_directed_at'] = label_encoder.fit_transform(train_data_df['emotion_in_tweet_is_directed_at'])
le_name_mapping_encoded_emotion_in_tweet_is_directed_at = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("le_name_mapping_encoded_emotion_in_tweet_is_directed_at ", le_name_mapping_encoded_emotion_in_tweet_is_directed_at)
train_data_df['encoded_is_there_an_emotion_directed_at_a_brand_or_product'] = label_encoder.fit_transform(train_data_df['is_there_an_emotion_directed_at_a_brand_or_product'])
le_name_mapping_encoded_is_there_an_emotion_directed_at_a_brand_or_product = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("le_name_mapping_encoded_is_there_an_emotion_directed_at_a_brand_or_product ", le_name_mapping_encoded_is_there_an_emotion_directed_at_a_brand_or_product)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
base_model = BertModel.from_pretrained("bert-base-uncased")
num_classes_label1 = 4  # For example, 3 classes
num_classes_label2 = 9  # For example, 5 classes

# Initialize the custom model
model = MultiTaskBERT(base_model, num_classes_label1, num_classes_label2)
# Assume `texts`, `labels1`, and `labels2` are your data lists
encodings, labels1, labels2 = preprocess_data(list(train_data_df['cleaned_tweet_text']), list(train_data_df['encoded_is_there_an_emotion_directed_at_a_brand_or_product']), list(train_data_df['encoded_emotion_in_tweet_is_directed_at']), tokenizer)

input_ids = torch.tensor(encodings['input_ids'])
attention_masks = torch.tensor(encodings['attention_mask'])
labels1 = torch.tensor(labels1)
labels2 = torch.tensor(labels2)

dataset = TensorDataset(input_ids, attention_masks, labels1, labels2)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Separate loss functions for each label
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask, label1, label2 = [item.to(device) for item in batch]
        
        optimizer.zero_grad()
        
        # Forward pass
        logits1, logits2 = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Compute individual losses
        loss1 = criterion1(logits1, label1)
        loss2 = criterion2(logits2, label2)
        
        # Combined loss
        loss = loss1 + loss2
        total_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

torch.save(model.state_dict(), PATH)

#####################################-----EVALUATION----######################################

model.eval()
predictions1, predictions2, true_labels1, true_labels2 = [], [], [], []

with torch.no_grad():
    for batch in dataloader:
        input_ids, attention_mask, label1, label2 = [item.to(device) for item in batch]
        
        logits1, logits2 = model(input_ids=input_ids, attention_mask=attention_mask)
        
        preds1 = torch.argmax(logits1, dim=1)
        preds2 = torch.argmax(logits2, dim=1)
        
        predictions1.extend(preds1.cpu().numpy())
        predictions2.extend(preds2.cpu().numpy())
        true_labels1.extend(label1.cpu().numpy())
        true_labels2.extend(label2.cpu().numpy())


# Label1 metrics
accuracy1 = accuracy_score(true_labels1, predictions1)
f1_score1 = f1_score(true_labels1, predictions1, average="weighted")

# Label2 metrics
accuracy2 = accuracy_score(true_labels2, predictions2)
f1_score2 = f1_score(true_labels2, predictions2, average="weighted")

print(f"Label1 - Accuracy: {accuracy1}, F1 Score: {f1_score1}")
print(f"Label2 - Accuracy: {accuracy2}, F1 Score: {f1_score2}")

