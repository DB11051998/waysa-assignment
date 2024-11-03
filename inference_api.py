from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from multi_model import MultiTaskBERT, preprocess_text

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
# Initialize FastAPI app
app = FastAPI()

# Load the BERT model and tokenizer
max_length = 128
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
base_model = BertModel.from_pretrained("bert-base-uncased")

# Label mappings
le_name_mapping_encoded_emotion_in_tweet_is_directed_at = {
    0: 'Android', 
    1: 'Android App', 
    2: 'Apple', 
    3: 'Google', 
    4: 'Other Apple product or service', 
    5: 'Other Google product or service', 
    6: 'iPad', 
    7: 'iPad or iPhone App', 
    8: 'iPhone'
}

le_name_mapping_encoded_is_there_an_emotion_directed_at_a_brand_or_product = {
    0: "I can't tell", 
    1: 'Negative emotion', 
    2: "No emotion toward brand or product", 
    3: 'Positive emotion'
}

# Initialize the custom model
num_classes_label1 = 4  # e.g., number of classes for the first label
num_classes_label2 = 9  # e.g., number of classes for the second label
PATH = "SAVED_MODEL_MULTICLASS.pt"
model = MultiTaskBERT(base_model, num_classes_label1, num_classes_label2)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')), strict=False)
model.eval()

# Define the request body structure
class TextRequest(BaseModel):
    text: str

# Route for prediction
@app.post("/predict")
async def predict_emotion(request: TextRequest):
    input_text = request.text
    cleaned_text = preprocess_text(input_text)
    encodings = tokenizer([cleaned_text], truncation=True, padding=True, max_length=max_length)
    
    # Prepare tensors
    input_ids = torch.tensor(encodings['input_ids'])
    attention_masks = torch.tensor(encodings['attention_mask'])

    with torch.no_grad():
        input_ids = input_ids.to('cpu')
        attention_mask = attention_masks.to('cpu')
        
        logits1, logits2 = model(input_ids=input_ids, attention_mask=attention_mask)
        preds1 = torch.argmax(logits1, dim=1).cpu().numpy()[0]
        preds2 = torch.argmax(logits2, dim=1).cpu().numpy()[0]
        
        # Map predictions to labels
        label1 = le_name_mapping_encoded_is_there_an_emotion_directed_at_a_brand_or_product[preds1]
        label2 = le_name_mapping_encoded_emotion_in_tweet_is_directed_at[preds2]

    return {"emotion": label1, "target": label2}

# To run the app, use: uvicorn main:app --reload
