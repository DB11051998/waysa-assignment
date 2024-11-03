import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel
import re
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords




class MultiTaskBERT(nn.Module):
    def __init__(self, base_model, num_classes_label1, num_classes_label2):
        super(MultiTaskBERT, self).__init__()
        self.bert = base_model
        # Define separate classification heads for each label
        self.classifier1 = nn.Linear(self.bert.config.hidden_size, num_classes_label1)
        self.classifier2 = nn.Linear(self.bert.config.hidden_size, num_classes_label2)
    
    def forward(self, input_ids, attention_mask):
        # Get BERT model outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]

        # Pass through each classifier head
        logits1 = self.classifier1(pooled_output)  # Output for label1
        logits2 = self.classifier2(pooled_output)  # Output for label2
        return logits1, logits2

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)       # Remove numbers
    tokens = word_tokenize(text)          # Tokenization
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)      # Return as a string