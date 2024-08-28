import os
import collections

import sentencepiece as spm
import sys
import json
import spacy



def main(input_path, dictionary_en_path, dictionary_de_path, lines_en_path, file_name,analyze_type):

    with open(dictionary_en_path, encoding='utf-8') as f:
        data = f.read()
        dictionary_en=json.loads(data)
    with open(dictionary_de_path, encoding='utf-8') as f: 
        data = f.read()
        dictionary_de=json.loads(data)
    with open(lines_en_path,encoding='utf-8') as f:
        lines_en=f.readlines()

    ref=input_path + ".ref"
    hyp=input_path + ".hyp"
    with open(ref, "r") as f:
        T_txt = f.readlines()
    with open(hyp, "r") as f:
        D_txt = f.readlines()
    print(len(lines_en), len(T_txt), len(D_txt))
    assert len(lines_en) == len(T_txt) == len(D_txt)

    nlp_de = spacy.load("de_core_news_sm") #en_core_web_sm" "de_core_news_sm"
    nlp = spacy.load("en_core_web_sm") #en_core_web_sm" "de_core_news_sm"

    un_aligned_words={}
    unrecog_lemma={}
    recog_lemma={}
    que_rarewords={}
    f_trans = open(file_name + "_translated_lemma.de", 'w',encoding='utf-8')    #write txt file
    f_untrans = open(file_name + "_un_translated_lemma.de", 'w', encoding = "utf-8")

    # Reading from the text file and converting its contents to a list of strings
    with open('rareword_in_train.txt', 'r', encoding='utf-8') as file:
        rareword_in_train = [line.strip() for line in file]  # Use list comprehension to read lines and remove newline characters

    for i, line_en in enumerate(lines_en):

        un_aligned_words[i] = []
        recog_lemma[i] = []
        unrecog_lemma[i] = []
        # detokenzie
        doc_en=nlp(line_en)
        pred_item_en=[token.lemma_.lower() for token in doc_en if not token.is_stop and not token.is_punct]

        line_T=T_txt[i]
        line_D=D_txt[i]
        assert len(line_T) > 0, len(line_D) > 0
        
        if analyze_type == 'ST':
            doc_D=nlp_de(line_D)
        elif analyze_type == 'ASR':
            doc_D=nlp(line_D)
        pred_item_D=[token.lemma_.lower() for token in doc_D if not token.is_stop and not token.is_punct]
        pred_item_D_new=" ".join(pred_item_D)

        que_rarewords[i] = []

        for j, item in enumerate(pred_item_en):

            if item not in dictionary_en:
                continue

            else:
                if len(dictionary_en[item]) == 1:
                    continue

                que_rarewords[i].append(item)
                if analyze_type == 'ST':
                    if item in dictionary_de:
                        items_de = dictionary_de[item]
                        sign = 0
                        for item_de in items_de:
                            if item_de in pred_item_D_new:
                                recog_lemma[i].append(item)
                                f_trans.write('item: '+item+'\n')
                                f_trans.write('item_de: '+item_de+'\n')
                                f_trans.write('T: '+line_T+'\n')
                                f_trans.write('D: '+line_D+'\n')
                                sign=1
                                break
                        if sign == 0:
                            unrecog_lemma[i].append(item)
                            f_untrans.write('item: '+item+'\n')
                            f_untrans.write('item_de: '+item_de+'\n')
                            f_untrans.write('T-en: '+line_en)
                            f_untrans.write('T-de: '+line_T+'\n')
                            f_untrans.write('D-de: '+line_D+'\n'+'\n')
                    else: #except:
                        un_aligned_words[i].append(item)
                elif analyze_type == 'ASR':
                    sign = 0
                    if item in pred_item_D_new:
                        recog_lemma[i].append(item)
                   
                        f_trans.write('item: '+item+'\n')
                        f_trans.write('item_de: '+item+'\n')
                        f_trans.write('T: '+line_T+'\n')
                        f_trans.write('D: '+line_D+'\n')
                        sign=1
                
                    else: #except:
                        un_aligned_words[i].append(item)  

                    if sign == 0:
                        unrecog_lemma[i].append(item)
                        f_untrans.write('item: '+item+'\n')
                        f_untrans.write('item_de: '+item+'\n')
                        f_untrans.write('T-en: '+line_en)
                        f_untrans.write('T-de: '+line_T+'\n')
                        f_untrans.write('D-de: '+line_D+'\n'+'\n')
    f_trans.close()
    f_untrans.close()

    total_rarewords = []
    total_sentence_rarewords_appear_in_train = 0
    total_sentence_rarewords_appear_not_in_train = 0

    for k,v in que_rarewords.items():
        total_rarewords = total_rarewords + v
    total_rarewords_set=set(total_rarewords)
    total_rarewords=list(total_rarewords_set)
    print("num of rare words is {}".format(len(total_rarewords))) #num of unrecognized lemma

    recognized_lemma = []
    # Open and read the file line by line
    with open(file_name+"_translated_lemma.de", 'r', encoding='utf-8') as file:
        for line in file:
        # Check if the line starts with 'item:'
            if line.startswith('item:'):
            # Extract the item name after 'item:' and strip any leading/trailing whitespace
                item_name = line.split('item:')[1].strip()
            # Append the item name to the list
                recognized_lemma.append(item_name)
    print("num of recognized lemma is {}".format(len(recognized_lemma))) #num of recognized word


    for item in recognized_lemma:
        if item not in rareword_in_train:
            total_sentence_rarewords_appear_not_in_train+=1
            
        else:
            total_sentence_rarewords_appear_in_train+=1

    print("num of rare words appear in train is {}".format(total_sentence_rarewords_appear_in_train)) #num of unrecognized lemma
    print("num of translated rare words appear not in train is {}".format(total_sentence_rarewords_appear_not_in_train)) #num of unrecognized lemma




    return D_txt,T_txt,que_rarewords



input_path=sys.argv[1] #/path/to/inference/result


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python script_name.py <english_file> <german_file>")
        sys.exit(1)

    input_path = sys.argv[1] 
    dictionary_en_path = sys.argv[2] 
    dictionary_de_path = sys.argv[3]
    lines_en_path=sys.argv[4]
    file_name=sys.argv[5]
    analyze_type=sys.argv[6]
    main(input_path, dictionary_en_path, dictionary_de_path, lines_en_path, file_name,analyze_type)
    