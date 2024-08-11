with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/tst_ex_new/txt/tst_ex_new.en') as f:
    tst_lines_new=f.readlines()
with open('/home/sli/DPR/downloads/data/fairseq/train.en') as f:
    train_lines=f.readlines()
with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/terminology/txt/terminology.en') as f:
    term_lines=f.readlines()

lines_tst_origin=[line.split(' <SEP> ')[1] for line in tst_lines_new]
lines_tst_example=[line.split(' <SEP> ')[0] for line in tst_lines_new]

import os



def get_audio_path(directory_path):
    # List all files in the directory and filter out those that are .wav files
    wav_files = [f for f in os.listdir(directory_path) if f.endswith('.wav')]
    # Define a function to extract the numeric part from the file name
    def extract_number(file_name):
        # Split the file name by '_' and take the second part, then remove the '.wav' extension and convert to int
        return int(file_name.split('_')[1].replace('.wav', ''))

    # Sort the list of files by the numeric part extracted from each file name
    sorted_list_by_number = sorted(wav_files, key=extract_number)
    audios_path_list = [directory_path+item for item in sorted_list_by_number]

    return audios_path_list

# Path to the directory containing the .wav files
directory_path = '/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/terminology/wav/'
term_audio_files=get_audio_path(directory_path)

tst_audio_files=get_audio_path('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/tst_ex_new/wav/')

import torch
import numpy
from dpr_sonar.data.biencoder_data_SONAR import BiEncoderPassage
from dpr_sonar.models import init_biencoder_components
from dpr_sonar.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger

from dpr_sonar.utils.data_utils_SONAR import Tensorizer
from dpr_sonar.utils.model_utils_SONAR import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)
from omegaconf import DictConfig, OmegaConf
import logging
# Configure the logging system
logging.basicConfig(level=logging.INFO)


cfg = OmegaConf.load("/home/sli/DPR_SONAR/conf/gen_embs_SONAR.yaml")
cfg.encoder= OmegaConf.load("/home/sli/DPR_SONAR/conf/encoder/sonar.yaml")
cfg = setup_cfg_gpu(cfg)
cfg.model_file='/export/data2/sli/checkpoints/sonar_dpr_finetune/dpr_biencoder.25' #'/export/data2/sli/checkpoints/sonar_dpr_finetune/dpr_biencoder.0'
saved_state = load_states_from_checkpoint(cfg.model_file)
set_cfg_params_from_state(saved_state.encoder_params, cfg)

tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)
encoder_ctx = encoder.ctx_model
encoder_question = encoder.question_model

encoder_ctx, _ = setup_for_distributed_mode(
        encoder_ctx,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
encoder_question, _ = setup_for_distributed_mode(
        encoder_question,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
encoder_ctx.eval()
encoder_question.eval()
model_to_load_ctx = get_model_obj(encoder_ctx)
model_to_load_question = get_model_obj(encoder_question)

ctx_state = {
        key: value for (key, value) in saved_state.model_dict["ctx_model"].items() 
    }
model_to_load_ctx.encoder.load_state_dict(ctx_state, strict=True)


question_state = {
        key: value for (key, value) in saved_state.model_dict["question_model"].items() 
    }
model_to_load_question.encoder.load_state_dict(question_state, strict=True)

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union
from fairseq2.data import Collater, StringLike
from sonar.inference_pipelines.utils import add_progress_bar, extract_sequence_batch
from fairseq2.data import SequenceData
import torch
from fairseq2.data.text import TextTokenizer, read_text
from fairseq2.data.data_pipeline import read_sequence
from SONAR.sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                           tokenizer="text_sonar_basic_encoder")

tokenizer=t2vec_model.tokenizer
tokenizer_encoder = tokenizer.create_encoder(lang="eng_Latn")

input=term_lines
sequence_batches=[]
for item in input:
        tokenized_data = [tokenizer_encoder(a) for a in [item]]
        max_seq_len=514
        def truncate(x: torch.Tensor) -> torch.Tensor:
                return x[:max_seq_len]
        # Truncate the data
        truncated_data = [truncate(item) for item in tokenized_data]

        batch=truncated_data
        # Pad the batches and prepare them for the model
        collated_batch = Collater(tokenizer.vocab_info.pad_idx)(batch)

        # Extract sequence batches and move to the device
        sequence_batch = extract_sequence_batch(collated_batch, 'cuda') 
        sequence_batches.append(sequence_batch)

batch_size=100
max=len(sequence_batches)
embeddings_ctx=[]
encoder_output= None
for i in range(99):
    # Write a log message
    logging.info(f'encode ctx {i}')
    with torch.no_grad():
        encoder_output=encoder_ctx(sequence_batches[i*batch_size:min(len(sequence_batches),(i+1)*batch_size)])
    if encoder_output.grad is not None:
        print("The tensor has a gradient.")
      
    else:
        print("The tensor does not have a gradient.")
    embeddings_ctx.append(encoder_output.cpu())
embeddings_ctx=torch.cat(embeddings_ctx,0)
print(len(embeddings_ctx))
print(embeddings_ctx.size())

input=lines_tst_origin
sequence_batches_tst=[]
for item in input:
        tokenized_data = [tokenizer_encoder(a) for a in [item]]
        max_seq_len=514
        def truncate(x: torch.Tensor) -> torch.Tensor:
                return x[:max_seq_len]
        # Truncate the data
        truncated_data = [truncate(item) for item in tokenized_data]

        batch=truncated_data
        # Pad the batches and prepare them for the model
        collated_batch = Collater(tokenizer.vocab_info.pad_idx)(batch)

        # Extract sequence batches and move to the device
        sequence_batch = extract_sequence_batch(collated_batch, 'cuda') 
        sequence_batches_tst.append(sequence_batch)

embeddings_question=[]
batch_size=100
max=len(sequence_batches)
encoder_output= None
for i in range(25):
    logging.info(f'question ctx {i}')
    with torch.no_grad():
        encoder_output=encoder_question(sequence_batches_tst[i*batch_size:min(len(sequence_batches_tst),(i+1)*batch_size)])
    embeddings_question.append(encoder_output.cpu())
embeddings_question=torch.cat(embeddings_question,0)

print(len(embeddings_question))
print(embeddings_question[0].shape)

# Normalize the vectors (dim=1 is the vector dimension)
normalized_tst = embeddings_question / embeddings_question.norm(dim=1, keepdim=True)
normalized_term = embeddings_ctx / embeddings_ctx.norm(dim=1, keepdim=True)

# Compute cosine similarity
cos_sim = torch.mm(normalized_tst, normalized_term.t())  # Shape: (2500, 9821)

# Find top 10 most similar for each in A
top_10_vals, top_10_indices = torch.topk(cos_sim, 10, dim=1)

# top_10_indices contains the indices of the top 10 most similar vectors in B for each vector in A
# top_10_vals contains the corresponding cosine similarity values

# If you want to convert them to Python lists for further processing
top_10_indices_list = top_10_indices.tolist()
top_10_vals_list = top_10_vals.tolist()

tst_term_pairs=[]
for i in range(len(top_10_indices_list)):
    tst_term_pairs.append([i,top_10_indices_list[i],top_10_vals_list[i]])
    
with open('/home/sli/DPR_SONAR/results/tst_term_pairs_dpr_sonar_finetune_q_n_p_text_text_10.txt','w') as f:
        for item in tst_term_pairs:
               f.write('{} \n'.format(item))

for i,pair in enumerate(tst_term_pairs):
    if i==0:
        with open('/home/sli/DPR_SONAR/results/tst_with_retrieved_example_dpr_sonar_finetune_q_n_p_text_text_1st_NN_3.txt','w') as g: #
            g.write(term_lines[pair[1][0]].strip()+' <SEP> '+lines_tst_origin[pair[0]])
        with open('/home/sli/DPR_SONAR/results/tst_with_retrieved_example_dpr_sonar_finetune_q_n_p_text_text_2st_NN_3.txt','w') as g: #
            g.write(term_lines[pair[1][1]].strip()+' <SEP> '+lines_tst_origin[pair[0]])
        with open('/home/sli/DPR_SONAR/results/tst_with_retrieved_example_dpr_sonar_finetune_q_n_p_text_text_3st_NN_3.txt','w') as g: #
            g.write(term_lines[pair[1][2]].strip()+' <SEP> '+lines_tst_origin[pair[0]])
    else:
        with open('/home/sli/DPR_SONAR/results/tst_with_retrieved_example_dpr_sonar_finetune_q_n_p_text_text_1st_NN_3.txt','a') as g: #
            g.write(term_lines[pair[1][0]].strip()+' <SEP> '+lines_tst_origin[pair[0]])
        with open('/home/sli/DPR_SONAR/results/tst_with_retrieved_example_dpr_sonar_finetune_q_n_p_text_text_2st_NN_3.txt','a') as g: #
            g.write(term_lines[pair[1][1]].strip()+' <SEP> '+lines_tst_origin[pair[0]])
        with open('/home/sli/DPR_SONAR/results/tst_with_retrieved_example_dpr_sonar_finetune_q_n_p_text_text_3st_NN_3.txt','a') as g: #
            g.write(term_lines[pair[1][2]].strip()+' <SEP> '+lines_tst_origin[pair[0]])
print("saved txt files")
with open('/home/sli/DPR_SONAR/results/tst_with_retrieved_example_dpr_sonar_finetune_q_n_p_text_text_1st_NN_3.txt',encoding='utf-8') as f:
    lines = f.readlines()
import json
with open('/home/sli/DPR/downloads/data/fairseq/rareword_terminology.txt') as f:
    data = f.read()
dictionary=json.loads(data)
import spacy
  
nlp=spacy.load("en_core_web_sm") 
examples=[]
sentences=[]
for line in lines:
    id_sep=line.find(' <SEP> ')
    example=line[:id_sep]
    sentence=line[id_sep+7:]
    examples.append(example)
    sentences.append(sentence)
sentence_rarewords={}
example_rarewords={}

for i in range(len(examples)):
    line_sentence=sentences[i]
    line_example=examples[i]
    
    sentence_rarewords[i]=[]
    example_rarewords[i]=[]
    
    doc_sentence=nlp(line_sentence)
    pred_item_sentence=[token.lemma_.lower() for token in doc_sentence if not token.is_stop and not token.is_punct]
    
    doc_example=nlp(line_example)
    pred_item_example=[token.lemma_.lower() for token in doc_example if not token.is_stop and not token.is_punct]
    pred_item_example_new=" ".join(pred_item_example)
    for j,item in enumerate(pred_item_sentence):
        try:
            dictionary[item]
            sentence_rarewords[i].append(item)

            if item in pred_item_example_new:
                example_rarewords[i].append(item)

        except:
            continue
    
total_rarewords=[]
for k,v in sentence_rarewords.items():
    total_rarewords=total_rarewords+v
    total_rarewords_set=set(total_rarewords)
    total_rarewords=list(total_rarewords)
    
    total_example_rarewords=[]
    for k,v in example_rarewords.items():
        total_example_rarewords=total_example_rarewords+v
    total_example_rarewords_set=set(total_example_rarewords)
    total_example_rarewords=list(total_example_rarewords)
    
print("num of rare words is {}".format(len(total_rarewords))) #num of unrecognized lemma
print("num of rare words in example is {}".format(len(total_example_rarewords))) #num of recognized word