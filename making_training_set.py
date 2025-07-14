import json

def main():
    with open('data/cleaned_dataset_medium_temp.jsonl', "r") as read_file:
        with open('data/cleaned_dataset_medium.jsonl', "w") as write_file:
            counter = 8326
            for line in read_file:
                if counter == 0:
                    break
                item = json.loads(line)
                write_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                write_file.flush()
                counter -= 1

if __name__ == "__main__":
    main()