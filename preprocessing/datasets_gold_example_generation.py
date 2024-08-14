#Construct datasets that prepend gold example to rare-word-tst split, rare-word-dev split, tst-COMMON split respectively, to form tst_ex split, dev_ex split, tst-COMMON_ex
#python preprocessing/datasets_gold_example_generation.py --dir ${MUSTC_ROOT}

#Construct train_ex split that each sentence in the reduced-train split is prepended sentences that contains the same sentence-level rare word from the reduced-train split
#python preprocessing/datasets_gold_example_generation.py --dir ${MUSTC_ROOT} --train True
from pydub import AudioSegment
import os
import yaml
import ast

def read_list_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Initialize an empty list to store the read items
        read_list = []
        
        # Iterate over each line
        for line in lines:
            # Use ast.literal_eval for safe evaluation of the string
            try:
                item = ast.literal_eval(line.strip())
                read_list.append(item)
            except ValueError as e:
                # Handle possible value error during literal_eval
                print(f"Error parsing line '{line.strip()}': {e}")
                
        return read_list

with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/tst-COMMON/txt/tst-COMMON.yaml') as f:
    my_dict_tst_COMMON = yaml.load(f,Loader=yaml.BaseLoader) 
with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/tst/txt/tst.yaml') as f:
    my_dict_tst = yaml.load(f,Loader=yaml.BaseLoader) 
with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/terminology/txt/terminology.yaml') as f:
    my_dict_term = yaml.load(f,Loader=yaml.BaseLoader)

with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/tst-COMMON/txt/tst-COMMON.en', encoding='utf-8') as f:
    lines_tst_COMMON_en = f.readlines()
with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/tst-COMMON/txt/tst-COMMON.de', encoding='utf-8') as f:
    lines_tst_COMMON_de = f.readlines()

with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/tst/txt/tst.en', encoding='utf-8') as f:
    lines_tst_en = f.readlines()
with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/tst/txt/tst.de', encoding='utf-8') as f:
    lines_tst_de = f.readlines()

with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/terminology/txt/terminology.en', encoding='utf-8') as f:
    lines_terminology_en = f.readlines()
with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/terminology/txt/terminology.de', encoding='utf-8') as f:
    lines_terminology_de = f.readlines()

wav_path='wav/'
txt_path='txt/'

###########################################################################################################
#CHANGE HERE
root_path_example = '/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/terminology/'
my_dict_example=my_dict_term   #where to retrieve example from
lines_en_example=lines_terminology_en
lines_de_example=lines_terminology_de


root_path = '/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/tst-COMMON/'
my_dict=my_dict_tst_COMMON #where to query
lines_en=lines_tst_COMMON_en
lines_de=lines_tst_COMMON_de

name='tst-COMMON_tmp_ex'
new_root_path = '/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/'+name+'/'

#file_path='/home/sli/DPR_SONAR/results/tst_term_pairs_dpr_sonar_finetune_q_n_p_text_text_freeze_15_10.txt'

###########################################################################################################
new_items=[]
import random
#retrieved_pair=read_list_from_file(file_path)

if not os.path.exists(new_root_path+wav_path): os.makedirs(new_root_path+wav_path) 
if not os.path.exists(new_root_path+txt_path): os.makedirs(new_root_path+txt_path) 
for i in range(len(lines_en)):
    sampled_index=random.randint(0,len(lines_en_example))
    duration_example=float(my_dict_example[sampled_index]['duration']) #start time of sentence
    offset_example=my_dict_example[sampled_index]['offset']#duration of sentence
    wav_example=my_dict_example[sampled_index]['wav']  #'ted_{}.wav'.format(item[1])  #audio file of sentence #
    # # #print(duration_exa)
    offset_example_ = float(offset_example)*1000 #Works in milliseconds
    duration_example_ = float(duration_example)*1000 
    newAudio_example = AudioSegment.from_wav(root_path_example+wav_path+wav_example)
    newAudio_example = newAudio_example[offset_example_:offset_example_+duration_example_]

            ##newAudio_silence = AudioSegment.from_wav(root_path_silence)     

    duration=float(my_dict[i]['duration']) #start time of sentence
    offset=my_dict[i]['offset']#duration of sentence
    wav=my_dict[i]['wav']  #'ted_{}.wav'.format(item[1])  #audio file of sentence #
    # # #print(duration_exa)
    offset_ = float(offset)*1000 #Works in milliseconds
    duration_ = float(duration)*1000 
    newAudio = AudioSegment.from_wav(root_path+wav_path+wav)
    newAudio = newAudio[offset_:offset_+duration_]

    newAudio = newAudio_example + newAudio
    newAudio.export(new_root_path+wav_path+'ted{}_{}.wav'.format(i,sampled_index), format="wav") #write audio data
    new_item={'duration':str(duration_example+duration),'offset':0.0,'rW':my_dict[i]['rW'],'uW':my_dict[i]['uW'],
                       'speaker_id':my_dict[i]['speaker_id'],'wav':'ted{}_{}.wav'.format(i,sampled_index)}
    new_items=new_items+[new_item]

    with open(new_root_path+txt_path+name+'.en', 'a',encoding='utf-8') as f:    #write txt file
            f.write(lines_en_example[sampled_index].strip()+' <SEP> '+lines_en[i])
    with open(new_root_path+txt_path+name+'.de', 'a',encoding='utf-8') as f:    #write txt file
            f.write(lines_de_example[sampled_index].strip()+' <SEP> '+lines_de[i])
    

with open(new_root_path+txt_path+name+'.yaml', 'a') as f:
    yaml.dump(new_items, f,default_flow_style=None,explicit_start=True)
           