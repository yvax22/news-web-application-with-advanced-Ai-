from transformers import AutoTokenizer, AutoModel
import torch

# تحميل نموذج AraBERT
model_name = "aubmindlab/bert-base-arabertv2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    # أخذ المتوسط عبر التوكنات (mean pooling)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding
