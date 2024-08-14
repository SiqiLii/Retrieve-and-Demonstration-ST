#take the train-split to evaluate how many rare words are there and where do they appear
#python preprocessing/rareword_analyze.py ${MUSTC_ROOT}/train/txt/train.en
import json
import spacy
import sys
import os

def process_file(file_path, output_dir):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines_en_train = file.readlines()

    nlp = spacy.load("en_core_web_sm")
    dictionary = {}
    non_rare_words = []

    for i, line in enumerate(lines_en_train):
        doc = nlp(line)
        vocabs = [token for token in doc if (not token.is_punct and not token.is_currency and not token.is_digit
                                             and not token.is_space and not token.text == "\n")]
        for vocab in vocabs:
            lemma = vocab.lemma_.lower()
            if lemma not in non_rare_words:
                try:
                    dictionary[lemma]
                except KeyError:
                    dictionary[lemma] = []
                if len(dictionary[lemma]) < 3:
                    dictionary[lemma].append(i)
                elif len(dictionary[lemma]) >= 3:
                    dictionary.pop(lemma)
                    non_rare_words.append(lemma)

    with open(os.path.join(output_dir, 'rareword_terminology.txt'), 'w') as f:
        f.write(json.dumps(dictionary))
    with open(os.path.join(output_dir, 'nonrareword_list.txt'), 'w') as f:
        f.write(json.dumps(non_rare_words))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = os.path.dirname(os.path.abspath(__file__))
    process_file(file_path, output_dir)
