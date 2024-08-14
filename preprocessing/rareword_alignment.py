#Then align rare words with their german translation
#python preprocessing/rareword_alignment.py --dir_en ${MUSTC_ROOT/train/txt/train.en} --dir_de ${MUSTC_ROOT/train/txt/train.de}
import spacy
import json
import sys
nlp=spacy.load("en_core_web_sm") 
nlp_de=spacy.load("de_core_news_sm") #de_dep_news_trf" "de_core_news_sm"

def read_word_pairs(file_path):
    word_pairs = {}
    with open('/home/sli/DPR/downloads/data/fairseq/nonrareword_list.txt', encoding='utf-8') as f:
        data = f.read()
    list_non_rareword=json.loads(data)
    with open(file_path, 'r', encoding='utf-8') as file:
        for i,line in enumerate(file):
            print(i)
            pairs = line.strip().split(' ')
            for pair in pairs:
                eng, ger = pair.split('<sep>')
                if eng in list_non_rareword:
                    continue
                nlp_eng=nlp(eng)
                lemmas_eng=[token.lemma_.lower() for token in nlp_eng if not token.is_stop and not token.is_punct]
                if len(lemmas_eng)==0:
                    continue
                lemma_eng=lemmas_eng[0]
                if lemma_eng in list_non_rareword:
                    continue

                nlp_ger=nlp_de(ger)
                lemmas_ger=[token.lemma_.lower() for token in nlp_ger if not token.is_stop and not token.is_punct]
                texts_ger=[token.text.lower() for token in nlp_ger if not token.is_stop and not token.is_punct]
                if len(lemmas_ger)==0:
                    continue
                
                lemma_ger=lemmas_ger[0]
                text_ger=texts_ger[0]

                if lemma_eng in word_pairs:
                    if lemma_ger not in word_pairs[lemma_eng]:
                        word_pairs[lemma_eng].add(lemma_ger)
                    if text_ger not in word_pairs[lemma_eng]:
                        word_pairs[lemma_eng].add(text_ger)
                else:
                    if lemma_ger!=lemma_eng:
                        word_pairs[lemma_eng] = {lemma_ger,text_ger}
                    else:
                        word_pairs[lemma_eng] = {lemma_ger}
    return word_pairs

def assign_translations(rare_words, word_pairs):
    translations = {}
    for word in rare_words:
        if word in word_pairs:
            translations[word] = list(word_pairs[word])
        else:
            translations[word] = ["No translation found"]
    return translations

def main(rareword_terminology_path,train_alignment_word_path):
    import json
    with open(rareword_terminology_path, encoding='utf-8') as f:
        data = f.read()
    dictionary=json.loads(data)
    
    rare_words = dictionary.keys()  # Example list, replace with your actual list
    
    word_pairs = read_word_pairs(train_alignment_word_path)
    print("finish reading word pairs")
    translations = assign_translations(rare_words, word_pairs)
    print("finish assigning translations")
    
    with open('rareword_terminology_de.txt', 'w', encoding='utf-8') as file:
        file.write(json.dumps(translations, ensure_ascii=False, indent=4))
    print("finish!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <english_file> <german_file>")
        sys.exit(1)

    rareword_terminology_path = sys.argv[1]
    train_alignment_word_path = sys.argv[2]
    main(rareword_terminology_path,train_alignment_word_path)
    