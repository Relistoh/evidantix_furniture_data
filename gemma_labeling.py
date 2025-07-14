import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="gemma3:12b")
template = (
    "Your task is to determine whether the following text refers to a **specific furniture product name**, "
    "not a general category or furniture type.\n\n"
    "Examples:\n"
    "Input: 'Malm Queen Bed Frame' → 1\n"
    "Input: 'Single Bed Bases' → 0\n"
    "Input: 'Oslo King Bed Frame' → 1\n"
    "Input: 'Bedroom Packages' → 0\n"
    "Input: 'Hampton 3-Seater Sofa' → 1\n"
    "Input: 'Bouclé Sofas' → 0\n"
    "Input: 'King Bed Head' → 0\n"
    "Input: 'IKEA Hemnes Bedside Table' → 1\n"
    "Input: 'Factory Buys 32cm Euro Top Mattress - King' → 1\n\n"
    "Now classify this text:\n"
    "{item_text}\n\n"
    "Only respond with 1 (if it's a specific product name) or 0 (if not). No explanation."
)

def labeling_with_gemma(product_dataset_file="data/product_dataset.jsonl"):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    with open(product_dataset_file, "r") as read_file:
        with open("data/labeled_dataset_gemma.jsonl", "w") as write_file:
            for line in read_file:
                item = json.loads(line)
                response = chain.invoke(
                    {"item_text": item['text']}
                )
                print(item['text'], response, sep='\n')
                label_str = response.strip()
                if label_str.find("1") != -1:
                    item["label"] = 1
                elif label_str.find("0") != -1:
                    item["label"] = 0
                else:
                    print(f"Unexpected output: {label_str}")
                    item["label"] = -1
                write_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                write_file.flush()

def main():
    labeling_with_gemma()

if __name__ == "__main__":
    main()