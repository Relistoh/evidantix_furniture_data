import json
import pandas as pd
import re


def reading_labeled_dataset(dataset_file, new_file):
    with open(dataset_file, "r") as read_file:
        with open(new_file, "w") as write_file:
            for line in read_file:
                item = json.loads(line)
                if item['label'] == -1 or item['label'] == 1:
                    write_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    write_file.flush()

def build_label_lookup(source_path):
    lookup = {}
    with open(source_path, "r", encoding="utf-8") as src:
        for line in src:
            item = json.loads(line)
            if item['text'] not in lookup:
                lookup[item["text"]] = item["label"]
    return lookup

def transfer_labels(source_path, target_path, output_path):
    label_lookup = build_label_lookup(source_path)

    with open(target_path, "r", encoding="utf-8") as target, \
            open(output_path, "w", encoding="utf-8") as out:

        for line in target:
            item = json.loads(line)
            if item["text"] in label_lookup:
                item["label"] = label_lookup[item["text"]]
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

def clean_dataset(df):
    df = df.drop_duplicates(subset=["text"])

    df = df[~df["text"].str.contains(r"\[\[.*?\]\]")]

    df = df[df["text"].str.len().between(4, 100)]

    tech_keywords = [
        "certificate", "net::", "ERR_", "javascript", "discount", "enable", "privacy",
        "protection", "warning", "enhanced", "issuer", "expires", "subject", "pantheon", "transparency"
    ]
    df = df[~df["text"].str.lower().str.contains('|'.join(tech_keywords))]

    ui_keywords = [
        "login", "cart", "menu", "view cart", "home", "privacy policy", "skip to content",
        "learn more", "review", "sign up", "read more", "email", "discount", "exclusive",
        "join", "subscribe", "stay updated", "read more reviews", "turn on", "proceed", "unsafe",
        "call us", "terms", "privacy policy", "help", "price", "pricing", "year", "warranty", "month"
    ]

    pattern = "|".join([re.escape(kw) for kw in ui_keywords])
    df = df[~df['text'].str.lower().str.contains(pattern)]

    df = df[df['text'].str.split().str.len() >= 2]

    df = df[df["text"].str.strip().str.match(r"^[\w\s\-,.&()'/]+$")]

    return df

def main():
    # transfer_labels('data/labeled_dataset.jsonl', 'data/product_dataset_with_urls.jsonl', 'data/labeled_dataset_with_urls.jsonl')
    # transfer_labels('data/labeled_dataset_gemma.jsonl', 'data/product_dataset_with_urls.jsonl', 'data/labeled_dataset_with_urls_gemma.jsonl')
    # reading_labeled_dataset('data/labeled_dataset_with_urls.jsonl', "data/labeled_dataset_gemma_1.jsonl")
    # reading_labeled_dataset('data/labeled_dataset_with_urls_gemma.jsonl', "data/labeled_dataset_1.jsonl")
    # transfer_labels('data/labeled_dataset_1.jsonl', 'data/cleaned_dataset_large.jsonl', 'data/cleaned_dataset_temp.jsonl')
    # reading_labeled_dataset('data/cleaned_dataset_temp.jsonl', 'data/cleaned_dataset_large_1.jsonl')
    transfer_labels('data/cleaned_dataset_large_1.jsonl', 'data/cleaned_dataset_large.jsonl', 'data/cleaned_dataset_medium_temp.jsonl')


    # df = pd.read_json("data/cleaned_dataset_temp.jsonl", lines=True)
    # cleaned_df = clean_dataset(df)
    # cleaned_df.to_json("data/cleaned_dataset_large.jsonl", orient="records", lines=True, force_ascii=False)

if __name__ == "__main__":
    main()
