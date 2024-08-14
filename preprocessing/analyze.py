import os
import collections

import sentencepiece as spm
import sys
import json
import spacy



def count_recognized_word_de(input_path, dictionary_en, dictionary_de, lines_en, file_name):
#   
#
    nlp_de = spacy.load("de_core_news_sm") #en_core_web_sm" "de_core_news_sm"
    nlp = spacy.load("en_core_web_sm") #en_core_web_sm" "de_core_news_sm"

    with open('/home/sli/DPR_SONAR/src/dpr_sonar/models/rareword_retrieved_in_audio_audio_same_speaker_removed.txt', 'r', encoding='utf-8') as file:
        rareword_in_retrieved = [line.strip() for line in file] 
    un_aligned_words={}
    unrecog_lemma={}
    recog_lemma={}
    que_rarewords={}
    recog_rarewords_in_retrieved={}
    f_trans = open(file_name + "_translated_lemma.de", 'w',encoding='utf-8')    #write txt file
    f_untrans = open(file_name + "_un_translated_lemma.de", 'w', encoding = "utf-8")

    for i, line_en in enumerate(lines_en):
#        if i in skipped:
#            continue

        # init counter
        un_aligned_words[i] = []
        recog_lemma[i] = []
        unrecog_lemma[i] = []
        recog_rarewords_in_retrieved[i] = []
        # detokenzie
        doc_en=nlp(line_en)
        pred_item_en=[token.lemma_.lower() for token in doc_en if not token.is_stop and not token.is_punct]

        line_T=T_txt[i]
        line_D=D_txt[i]
        assert len(line_T) > 0, len(line_D) > 0
#        if len(line_T)==0:
#            continue
        
        doc_D=nlp_de(line_D)

        pred_item_D=[token.lemma_.lower() for token in doc_D if not token.is_stop and not token.is_punct]
        pred_item_D_new=" ".join(pred_item_D)

        que_rarewords[i] = []

        for j, item in enumerate(pred_item_en):
#            try:
#                dictionary_en[item]
            if item not in dictionary_en:
                #print(f"{item} not in dictionary_en!")
                continue

            else:
                if len(dictionary_en[item]) == 1:
#                     # skip entries occurring only once
                    continue

                que_rarewords[i].append(item)

                if item in dictionary_de:
                    items_de = dictionary_de[item]
                    sign = 0
                    for item_de in items_de:
                        if item_de in pred_item_D_new:
                            recog_lemma[i].append(item)
                            if item in rareword_in_retrieved:
                                recog_rarewords_in_retrieved[i].append(item)
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
    f_trans.close()
    f_untrans.close()
#            except:
#                continue

    total_rarewords = []
    total_sentence_rarewords_appear_in_train = 0
    total_sentence_rarewords_appear_not_in_train = 0
    total_sentence_rarewords_appear_not_in_matched=0
    total_sentence_rarewords_appear_matched=0
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


    # Reading from the text file and converting its contents to a list of strings
    with open('rareword_in_train.txt', 'r', encoding='utf-8') as file:
        rareword_in_train = [line.strip() for line in file]  # Use list comprehension to read lines and remove newline characters

    with open('rareword_matched.txt', 'r', encoding='utf-8') as file:
        rareword_matched = [line.strip() for line in file]  # Use list comprehension to read lines and remove newline characters

    for item in recognized_lemma:
        if item not in rareword_in_train:
            total_sentence_rarewords_appear_not_in_train+=1
            
        else:
            total_sentence_rarewords_appear_in_train+=1

    print("num of rare words appear in train is {}".format(total_sentence_rarewords_appear_in_train)) #num of unrecognized lemma
    print("num of translated rare words appear not in train is {}".format(total_sentence_rarewords_appear_not_in_train)) #num of unrecognized lemma

    
    matched_lemma=[]
    for item in recognized_lemma:
        if item in rareword_matched:
            matched_lemma.append(item)
    matched_lemma=list(set(matched_lemma))
    print("num of rare words appear in matched is {}".format(len(matched_lemma))) #num of unrecognized lemma

    for item in matched_lemma:
        if item in rareword_in_train:
            total_sentence_rarewords_appear_matched+=1
        else:
            total_sentence_rarewords_appear_not_in_matched+=1
    print("num of translated rare words appear in matched not in train is {}".format(total_sentence_rarewords_appear_not_in_matched)) #num of unrecognized lemma
    print("num of translated rare words appear in matched in train is {}".format(total_sentence_rarewords_appear_matched)) #num of unrecognized lemma

    total_rarewords=[]
    for k,v in recog_rarewords_in_retrieved.items():
        total_rarewords = total_rarewords + v
    total_rarewords_set=set(total_rarewords)
    total_rarewords=list(total_rarewords_set)
    print("num of rare words appear in retrieved is {}".format(len(total_rarewords))) #num of unrecognized lemma


    return D_txt,T_txt,que_rarewords



input_path=sys.argv[1] #'/home/sli/text-to-text-data_ex/tst.en-de.decode.log' /home/sli/term_acc_cal/outs/mustc_st_en_de_s2t_transformer_s_ex_smallLRd.d0.2.tst-COMMON_tmp_ex_st.checkpoint_avg_last10.bsz1.full.out' /home/sli/term_acc_cal/outs/mustc_st_en_de_s2t_transformer_s_term.tst-COMMON_st.checkpoint_avg10_30k.maxtok50000.full.out' mustc_st_en_de_s2t_transformer_s_term.tst-COMMON_st.checkpoint_avg10_30k.maxtok50000.full.out
#'/export/data2/dliu/data/st/mustc_st_en_de_s2t_transformer_s_term.tst_st.checkpoint_avg10_30k.maxtok50000.full.out'
ref=input_path + ".ref"
hyp=input_path + ".hyp"


with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/tst-COMMON/txt/tst-COMMON.en',encoding='utf-8') as f:
    lines_tst_common=f.readlines()
with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/tst/txt/tst.en',encoding='utf-8') as f:
    lines_tst=f.readlines()
with open(ref, "r") as f:
    T_txt = f.readlines()
with open(hyp, "r") as f:
    D_txt = f.readlines()
print(len(lines_tst), len(T_txt), len(D_txt))
assert len(lines_tst) == len(T_txt) == len(D_txt)

with open('/home/sli/DPR/downloads/data/fairseq/rareword_terminology.txt', encoding='utf-8') as f:
    data = f.read()
    dictionary=json.loads(data)
with open('/home/sli/DPR/downloads/data/fairseq/rareword_terminology_de.txt', encoding='utf-8') as f: 
    #/home/sli/term_acc_cal/tst_dictionary_de_tmp_0220_19.txt
    data = f.read()
    dictionary_de=json.loads(data)

#D_results,T_results,que_rarewords = 
count_recognized_word_de(input_path, dictionary_en=dictionary, dictionary_de=dictionary_de, lines_en=lines_tst, file_name="lines_output_st_tst_audio_audio_transformer") #lines_en=lines_tst
#print(que_rarewords)
