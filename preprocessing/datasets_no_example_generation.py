#Construct datasets that prepend gold example to rare-word-tst split, rare-word-dev split, tst-COMMON split respectively, to form tst_ex split, dev_ex split, tst-COMMON_ex
#python preprocessing/datasets_gold_example_generation.py --dir ${MUSTC_ROOT}

#Construct train_ex split that each sentence in the reduced-train split is prepended sentences that contains the same sentence-level rare word from the reduced-train split
#python preprocessing/datasets_gold_example_generation.py --dir ${MUSTC_ROOT} --train True
from pydub import AudioSegment
import os
import yaml
import ast
import random
import argparse
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
    
def process_audio(root_path_origin, yaml_origin_path, lines_en_origin_path, lines_de_origin_path, new_root_path, name, file_path_index_list, 
                  wav_path='wav/', txt_path='txt/'):

    
    with open(yaml_origin_path) as f:
        my_dict = yaml.load(f,Loader=yaml.BaseLoader)
    with open(lines_en_origin_path, encoding='utf-8') as f:
        lines_en = f.readlines()
    with open(lines_de_origin_path, encoding='utf-8') as f:
        lines_de = f.readlines()
    

    new_items=[]
    
    index_list=read_list_from_file(file_path_index_list)


    if not os.path.exists(new_root_path+wav_path): os.makedirs(new_root_path+wav_path) 
    if not os.path.exists(new_root_path+txt_path): os.makedirs(new_root_path+txt_path) 

    for i in range(len(index_list)):
        sampled_index=index_list[i][1][0]  #random.randint(0,len(lines_en_example))
        duration=float(my_dict[sampled_index]['duration']) #start time of sentence
        offset=my_dict[sampled_index]['offset']#duration of sentence
        wav=my_dict[sampled_index]['wav']  #'ted_{}.wav'.format(item[1])  #audio file of sentence #
        # # #print(duration_exa)
        offset_ = float(offset)*1000 #Works in milliseconds
        duration_ = float(duration)*1000 
        newAudio = AudioSegment.from_wav(root_path_origin+wav_path+wav)
        newAudio = newAudio[offset_:offset_+duration_]

        newAudio.export(new_root_path+wav_path+'ted_{}.wav'.format(sampled_index), format="wav") #write audio data
        new_item={'duration':str(duration),'offset':0.0,'rW':my_dict[i]['rW'],'uW':my_dict[i]['uW'],
                       'speaker_id':my_dict[i]['speaker_id'],'wav':'ted_{}.wav'.format(sampled_index)}
        new_items=new_items+[new_item]

        with open(new_root_path+txt_path+name+'.en', 'a',encoding='utf-8') as f:    #write txt file
            f.write(lines_en[sampled_index].strip()+' <SEP> '+lines_en[i])
        with open(new_root_path+txt_path+name+'.de', 'a',encoding='utf-8') as f:    #write txt file
            f.write(lines_de[sampled_index].strip()+' <SEP> '+lines_de[i])
    

    with open(new_root_path+txt_path+name+'.yaml', 'a') as f:
        yaml.dump(new_items, f,default_flow_style=None,explicit_start=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio and text files to create new dataset splits.")
    parser.add_argument("--path_origin_yaml", type=str, required=True, help="Path to the origin yaml data.")
    parser.add_argument("--path_origin_en", type=str, required=True, help="Path to the origin en data.")
    parser.add_argument("--path_origin_de", type=str, required=True, help="Path to the origin de data.")
    parser.add_argument("--root_path_origin", type=str, required=True, help="Path to the directory containing target data.")
    parser.add_argument("--new_root_path", type=str, required=True, help="Path to the directory for saving new dataset splits.")
    parser.add_argument("--name", type=str, required=True, help="Name of the new dataset split.")
    parser.add_argument("--file_path_index_list", type=str, required=True, help="Path to the file containing the index pair.")

    args = parser.parse_args()

    process_audio(root_path_origin=args.root_path_origin, 
                  yaml_origin_path= args.path_origin_yaml,
                  lines_en_origin_path=args.path_origin_en,
                  lines_de_origin_path=args.path_origin_de,
                  new_root_path=args.new_root_path, 
                  name=args.name,   
                  file_path_index_list=args.file_path_index_list,
    )




           