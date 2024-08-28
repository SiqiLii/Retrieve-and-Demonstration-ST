#Use analyzed rare word information to construct reduced-train split, rare-word-tst split, rare-word-dev split, rare-word-pool: named train, tst, dev, terminology respectively
#python preprocessing/datasets_gold_example_generation.py ${MUSTC_ROOT} ${path_to_rareword_terminology_txt} ${path_to_rareword_terminology_de_txt}
import json
import spacy
import random
nlp=spacy.load("en_core_web_sm") 
nlp_de=spacy.load("de_core_news_sm") 

path_to_rareword_terminology_txt='/home/sli/DPR/downloads/data/fairseq/rareword_terminology.txt'
path_to_rareword_terminology_de_txt='/home/sli/DPR/downloads/data/fairseq/rareword_terminology_de.txt'
path_to_train_en='/home/sli/DPR/downloads/data/fairseq/train.en'

with open(path_to_rareword_terminology_txt) as f:
    data = f.read()
dictionary_en=json.loads(data)


with open(path_to_rareword_terminology_de_txt, encoding='utf-8') as f: #/home/sli/term_acc_cal/tst_dictionary_de_tmp_0220_19.txt
    data = f.read()
dictionary_de=json.loads(data)


with open(path_to_train_en, encoding='utf-8') as f:
    lines_train_en = f.readlines()


def assign_index(index_set,train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list):
    if len(index_set)==2:
        index_0,index_1=index_set
        if (index_0 in rare_word_dev_tst_list or index_0 in rare_word_pool_list) and (index_1 in rare_word_dev_tst_list or index_1 in rare_word_pool_list):
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_0 in rare_word_dev_tst_list) and (index_1 in train_reduced_list):
            rare_word_pool_list.append(index_1)
            train_reduced_list.remove(index_1)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_0 in rare_word_pool_list) and (index_1 in train_reduced_list):
            rare_word_dev_tst_list.append(index_1)
            train_reduced_list.remove(index_1)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_1 in rare_word_dev_tst_list) and (index_0 in train_reduced_list):
            rare_word_pool_list.append(index_0)
            train_reduced_list.remove(index_0)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_1 in rare_word_pool_list) and (index_0 in train_reduced_list):
            rare_word_dev_tst_list.append(index_0)
            train_reduced_list.remove(index_0)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        else:
            rare_word_pool_list.append(index_0)
            rare_word_dev_tst_list.append(index_1)
            train_reduced_list.remove(index_0)
            train_reduced_list.remove(index_1)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
    if len(index_set)==3:
        index_0,index_1,index_2=index_set
        if (index_0 not in train_reduced_list) and (index_1 not in train_reduced_list) and (index_2 not in train_reduced_list):
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_0 in rare_word_dev_tst_list) and (index_1 in train_reduced_list) and (index_2 in train_reduced_list):
            rare_word_pool_list.append(index_1)
            train_reduced_list.remove(index_1)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_0 in rare_word_pool_list) and (index_1 in train_reduced_list) and (index_2 in train_reduced_list):
            rare_word_dev_tst_list.append(index_1)
            train_reduced_list.remove(index_1)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_1 in rare_word_dev_tst_list) and (index_0 in train_reduced_list) and (index_2 in train_reduced_list):
            rare_word_pool_list.append(index_0)
            train_reduced_list.remove(index_0)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_1 in rare_word_pool_list) and (index_0 in train_reduced_list) and (index_2 in train_reduced_list):
            rare_word_dev_tst_list.append(index_0)
            train_reduced_list.remove(index_0)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_2 in rare_word_dev_tst_list) and (index_0 in train_reduced_list) and (index_1 in train_reduced_list):
            rare_word_pool_list.append(index_0)
            train_reduced_list.remove(index_0)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_2 in rare_word_pool_list) and (index_0 in train_reduced_list) and (index_1 in train_reduced_list):
            rare_word_dev_tst_list.append(index_0)
            train_reduced_list.remove(index_0)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        
        elif (index_0 in rare_word_dev_tst_list) and (index_1 in rare_word_dev_tst_list) and (index_2 in train_reduced_list):
            rare_word_pool_list.append(index_2)
            train_reduced_list.remove(index_2)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_0 in rare_word_pool_list) and (index_1 in rare_word_pool_list) and (index_2 in train_reduced_list):
            rare_word_dev_tst_list.append(index_2)
            train_reduced_list.remove(index_2)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_0 in rare_word_dev_tst_list) and (index_2 in rare_word_dev_tst_list) and (index_1 in train_reduced_list):
            rare_word_pool_list.append(index_1)
            train_reduced_list.remove(index_1)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_0 in rare_word_pool_list) and (index_2 in rare_word_pool_list) and (index_1 in train_reduced_list):
            rare_word_dev_tst_list.append(index_1)
            train_reduced_list.remove(index_1)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_1 in rare_word_dev_tst_list) and (index_2 in rare_word_dev_tst_list) and (index_0 in train_reduced_list):
            rare_word_pool_list.append(index_0)
            train_reduced_list.remove(index_0)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_1 in rare_word_pool_list) and (index_2 in rare_word_pool_list) and (index_0 in train_reduced_list):
            rare_word_dev_tst_list.append(index_0)
            train_reduced_list.remove(index_0)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        elif (index_0 in train_reduced_list) and (index_1 in train_reduced_list) and (index_2 in train_reduced_list):
            rare_word_pool_list.append(index_2)
            rare_word_dev_tst_list.append(index_1)
            train_reduced_list.remove(index_2)
            train_reduced_list.remove(index_1)
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
        else:
            return train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list
def select_and_split_items(input_list, num_select):
    # Randomly select 25 items from the input list
    selected_items = random.sample(input_list, num_select)
    
    # Create the remaining items list
    remaining_items = [item for item in input_list if item not in selected_items]
    
    return selected_items, remaining_items
            
train_list=[i for i in range(len(lines_train_en))]
train_reduced_list=train_list.copy()
rare_word_dev_tst_list=[]
rare_word_pool_list=[]
rare_word_visited={}

for i in train_list:
    #assert len(train_reduced_list)+len(rare_word_dev_tst_list)+len(rare_word_pool_list)==len(lines_train_en)
    line=lines_train_en[i]
    doc=nlp(line)
    line_item=[token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    for j, item in enumerate(line_item):
            if item not in dictionary_en:
                continue
            else:
                if len(set(dictionary_en[item])) == 1:
                    continue
            
                if len(set(dictionary_en[item])) == 2:
                    index_0,index_1=set(dictionary_en[item])
                    train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list=assign_index(set(dictionary_en[item]),train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list)
                    train_reduced_list=list(set(train_reduced_list))
                    rare_word_dev_tst_list=list(set(rare_word_dev_tst_list))
                    rare_word_pool_list=list(set(rare_word_pool_list))
                    assert len(train_reduced_list)+len(rare_word_dev_tst_list)+len(rare_word_pool_list)==len(lines_train_en)
                    #break
                if len(set(dictionary_en[item])) == 3:
                    index_0,index_1,index_2=set(dictionary_en[item])
                    train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list=assign_index(set(dictionary_en[item]),train_reduced_list,rare_word_dev_tst_list,rare_word_pool_list)
                    train_reduced_list=list(set(train_reduced_list))
                    rare_word_dev_tst_list=list(set(rare_word_dev_tst_list))
                    rare_word_pool_list=list(set(rare_word_pool_list))
                    assert len(train_reduced_list)+len(rare_word_dev_tst_list)+len(rare_word_pool_list)==len(lines_train_en)
                    #break

rare_word_tst_list, rare_word_dev_list = select_and_split_items(rare_word_dev_tst_list,num_select=2500)
print(len(train_reduced_list),len(rare_word_dev_tst_list),len(rare_word_pool_list))
assert len(train_reduced_list)+len(rare_word_dev_tst_list)+len(rare_word_pool_list)==len(lines_train_en)


with open('train_reduced_list.txt', 'w') as file:
    for number in train_reduced_list:
        file.write(f"{number}\n")

#with open('train_reduced_list.txt', 'r') as file:
#    train_reduced_list_read = [int(line.strip()) for line in file]

with open('rare_word_dev_list.txt', 'w') as file:
    for number in rare_word_dev_list:
        file.write(f"{number}\n")



with open('rare_word_tst_list.txt', 'w') as file:
    for number in rare_word_tst_list:
        file.write(f"{number}\n")


with open('rare_word_pool_list.txt', 'w') as file:
    for number in rare_word_pool_list:
        file.write(f"{number}\n")

def get_example_list(input_list, rare_word_pool_list, dictionary_en,lines):
    example_list = []
    for i in input_list:
        line = lines[i]
        doc = nlp(line)
        line_item = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        for j, item in enumerate(line_item):
            if item in dictionary_en:
                for k in rare_word_pool_list:
                    rare_word_pool_line = lines[k]
                    rare_word_pool_doc = nlp(rare_word_pool_line)
                    rare_word_pool_line_item = [token.lemma_.lower() for token in rare_word_pool_doc if not token.is_stop and not token.is_punct]
                    if item in rare_word_pool_line_item:
                        example_list.append([i,[k]])
                        break
                break
            else:
                continue
    return example_list

train_reduced_example_list=get_example_list(train_reduced_list,rare_word_pool_list,dictionary_en,lines_train_en)
with open('train_reduced_example_list.txt','w') as f:
        for item in train_reduced_example_list:
               f.write('{} \n'.format(item))

rare_word_dev_example_list=get_example_list(rare_word_dev_list,rare_word_pool_list,dictionary_en,lines_train_en)
with open('rare_word_dev_example_list.txt','w') as f:
        for item in rare_word_dev_example_list:
               f.write('{} \n'.format(item))
               rare_word_dev_example_list=get_example_list(train_reduced_list,rare_word_pool_list,dictionary_en,lines_train_en)

rare_word_tst_example_list=get_example_list(rare_word_tst_list,rare_word_pool_list,dictionary_en, lines_train_en)
with open('rare_word_tst_example_list.txt','w') as f:
        for item in rare_word_tst_example_list:
               f.write('{} \n'.format(item))

