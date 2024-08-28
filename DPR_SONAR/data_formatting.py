import json
import spacy
import sys
nlp=spacy.load("en_core_web_sm") 


def main(rareword_terminology_path,train_en_path,dev_ex_new_en_path):
    with open(rareword_terminology_path) as f:
        data = f.read()
    dictionary=json.loads(data)
    with open(train_en_path) as f:
        train_lines=f.readlines()
    with open(dev_ex_new_en_path) as f:
        dev_lines_new=f.readlines()

    dataset=[]
    rareword_list=list(dictionary.keys())
    train_lines_len=len(train_lines)
    list_sent=list(range(train_lines_len))
    for rareword in rareword_list:
        if len(dictionary[rareword])==1:
            continue
        data_example={}
        data_example['dataset']='fairseq'
        data_example['question']=train_lines[dictionary[rareword][0]]
        data_example['answers']=[rareword]

        data_example['positive_ctxs']=[]
        for sentence_idx in dictionary[rareword][1:]:
            positive_ctx={}
            positive_ctx['title']=''
            positive_ctx['text']=train_lines[sentence_idx]
            positive_ctx['score']=1000
            positive_ctx['title_score']=0
            positive_ctx['passage_id']=sentence_idx
            data_example['positive_ctxs'].append(positive_ctx)
    
        sent_select_from = [i for i in list_sent if i not in dictionary[rareword]]
        import random
        random_selected_sent=random.sample(sent_select_from, 50)
        for sentence_idx in random_selected_sent:
            positive_ctx={}
            positive_ctx['title']=''
            positive_ctx['text']=train_lines[sentence_idx]
            positive_ctx['score']=0
            positive_ctx['title_score']=0
            positive_ctx['passage_id']=sentence_idx
            data_example['positive_ctxs'].append(positive_ctx)
    
    
        data_example['negative_ctxs']=[]
    
        data_example['hard_negative_ctxs']=[]
        dataset.append(data_example)

    with open('data_train_audio_new_2.json', 'w') as f:
        json.dump(dataset, f)



    dataset_tst=[]
    sentence_rarewords={}
    example_rarewords={}
    for i,line in enumerate(dev_lines_new):
        sentence_rarewords[i]=[]
        example_rarewords[i]=[]
        id_sep=line.find(' <SEP> ')
        example=line[:id_sep]
        sentence=line[id_sep+7:]
    
        doc_sentence=nlp(sentence)
        pred_item_sentence_lemma=[token.lemma_.lower() for token in doc_sentence if not token.is_stop and not token.is_punct]
        pred_item_sentence=[token.text for token in doc_sentence if not token.is_stop and not token.is_punct]
        assert len(pred_item_sentence_lemma)==len(pred_item_sentence)
    
        for j,item in enumerate(pred_item_sentence_lemma):
            try:
                dictionary[item]
                #print(item)
                data_example={}
                data_example['dataset']='fairseq'
                data_example['question']=sentence
                data_example['answers']=[pred_item_sentence[j]]

                data_example['positive_ctxs']=[]
            
                positive_ctx={}
                positive_ctx['title']=''
                positive_ctx['text']=example
                positive_ctx['score']=1000
                positive_ctx['title_score']=0
                positive_ctx['passage_id']=i
                data_example['positive_ctxs'].append(positive_ctx)
                data_example['negative_ctxs']=[]
    
                data_example['hard_negative_ctxs']=[]
    
    
                dataset_tst.append(data_example)
                
            except:
                continue
    with open('data_dev_audio.json', 'w') as f:
        json.dump(dataset_tst, f)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script_name.py <english_file> <german_file>")
        sys.exit(1)

    rareword_terminology_path = sys.argv[1]
    train_en_path = sys.argv[2]
    dev_ex_new_en_path = sys.argv[3]
    main(rareword_terminology_path,train_en_path,dev_ex_new_en_path)
    