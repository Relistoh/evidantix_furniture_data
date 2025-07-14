import json
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from extracting_data import scrape_website, extract_text_nodes_in_order, extract_body_content
from cleaning_dataset import clean_dataset

model = AutoModelForSequenceClassification.from_pretrained("./models/product_classifier_cleaned_medium")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

st.title("AI Furniture Extractor")
url = st.text_input("Enter Website URL")

if st.button("Extract Furniture Data"):
    if url:
        with st.spinner("Scraping and analyzing website..."):
            try:
                html = scrape_website(url)
                body = extract_body_content(html)
                nodes = extract_text_nodes_in_order(body, url)

                with open('website_data.jsonl', 'w', encoding='utf-8') as write_file:
                    for item in nodes:
                        write_file.write(json.dumps(item, ensure_ascii=False) + "\n")

                df = pd.read_json("website_data.jsonl", lines=True)
                cleaned_df = clean_dataset(df)

                results = []

                for text in cleaned_df['text'].tolist():
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        prediction = torch.argmax(outputs.logits, dim=1).item()
                    if prediction == 1:
                        print(text)
                        results.append(text)

                furniture_df = pd.DataFrame(results, columns=["Furniture Product"])

                if not furniture_df.empty:
                    st.success(f"Found {len(furniture_df)} furniture products!")
                    st.subheader("Extracted Furniture Products")
                    st.dataframe(furniture_df)
                else:
                    st.warning("No furniture products were detected.")

            except Exception as e:
                st.error(f"Error while processing the URL: {type(e).__name__} – {e}")
