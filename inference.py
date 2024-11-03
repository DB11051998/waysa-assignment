import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from multi_model import MultiTaskBERT,preprocess_text
# le_name_mapping_encoded_emotion_in_tweet_is_directed_at={'Android': 0, 'Android App': 1, 'Apple': 2, 'Google': 3, 'Other Apple product or service': 4, 'Other Google product or service': 5, 'iPad': 6, 'iPad or iPhone App': 7, 'iPhone': 8}
le_name_mapping_encoded_emotion_in_tweet_is_directed_at={
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

# le_name_mapping_encoded_is_there_an_emotion_directed_at_a_brand_or_product={"I can't tell": 0, 'Negative emotion': 1, 'No emotion toward brand or product': 2, 'Positive emotion': 3}
le_name_mapping_encoded_is_there_an_emotion_directed_at_a_brand_or_product = {
    0:"I can't tell", 
    1:'Negative emotion' , 
    2:"No emotion toward brand or product", 
    3:'Positive emotion'
}

max_length=128
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
base_model = BertModel.from_pretrained("bert-base-uncased")

input_text=input()

cleaned_text=preprocess_text(input_text)
list_cleaned_text=[cleaned_text]
encodings = tokenizer(list_cleaned_text, truncation=True, padding=True, max_length=max_length)

num_classes_label1 = 4  # For example, 3 classes
num_classes_label2 = 9  # For example, 5 classes
PATH="SAVED_MODEL_MULTICLASS.pt"
# Initialize the custom model
model = MultiTaskBERT(base_model, num_classes_label1, num_classes_label2)
model.load_state_dict(torch.load(PATH, weights_only=True))

input_ids = torch.tensor(encodings['input_ids'])
attention_masks = torch.tensor(encodings['attention_mask'])
dataset = TensorDataset(input_ids, attention_masks)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    for batch in dataloader:
        input_ids, attention_mask = [item.to(device) for item in batch]
        
        logits1, logits2 = model(input_ids=input_ids, attention_mask=attention_mask)
        
        preds1 = torch.argmax(logits1, dim=1)
        preds2 = torch.argmax(logits2, dim=1)

        ## convert the preds to classes names using the dictionary, which was used to label encode.
        print(preds1,preds2)
        print(le_name_mapping_encoded_emotion_in_tweet_is_directed_at[preds2.cpu().numpy()[0]],le_name_mapping_encoded_is_there_an_emotion_directed_at_a_brand_or_product[preds1.cpu().numpy()[0]])