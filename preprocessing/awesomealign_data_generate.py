import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import sys
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def write_alignment_file(english_sentences, german_sentences, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for eng_sentence, ger_sentence in zip(english_sentences, german_sentences):
            eng_tokens = word_tokenize(eng_sentence)
            ger_tokens = word_tokenize(ger_sentence)
            file.write(' '.join(eng_tokens) + ' ||| ' + ' '.join(ger_tokens) + '\n')

def main(english_file, german_file):

    english_sentences = read_file(english_file)
    german_sentences = read_file(german_file)

    output_file_example = 'train_ende.src-tgt'
    
    
    
    if len(english_sentences) != len(german_sentences):
        raise ValueError("The number of sentences in the English file and German file must be the same.")
    
    write_alignment_file(english_sentences, german_sentences, output_file_example)
    print(f"Alignment file written to {output_file_example}")
 

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <english_file> <german_file>")
        sys.exit(1)

    english_file = sys.argv[1]
    german_file = sys.argv[2]
    main(english_file, german_file)