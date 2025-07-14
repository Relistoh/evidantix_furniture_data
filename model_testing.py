from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from extracting_data import scrape_website, extract_text_nodes_in_order

model = AutoModelForSequenceClassification.from_pretrained("./models/product_classifier_llama3_cleaned_tiny")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

html_content = scrape_website("https://www.factorybuys.com.au/products/euro-top-mattress-king")
nodes = extract_text_nodes_in_order(html_content, "https://www.factorybuys.com.au/products/euro-top-mattress-king")
window_size = 2

for i in range(len(nodes)):
    full_text = nodes[i]['text']
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    if prediction == 1:
        print(full_text, " -> PRODUCT")