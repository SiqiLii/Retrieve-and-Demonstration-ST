import os
import scipy.io.wavfile as wav
import torch



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
tst_audio_files=get_audio_path('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/tst/wav/')

with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/tst_ex_new/txt/tst_ex_new.en') as f:
    tst_lines_new=f.readlines()
with open('/home/sli/DPR/downloads/data/fairseq/train.en') as f:
    train_lines=f.readlines()
with open('/export/data2/sli/data/MuST-C_synthesized/de/en-de/data/terminology/txt/terminology.en') as f:
    term_lines=f.readlines()

lines_tst_origin=[line.split(' <SEP> ')[1] for line in tst_lines_new]
lines_tst_example=[line.split(' <SEP> ')[0] for line in tst_lines_new]

from SONAR.sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
s2vec_model = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_eng")
import torchaudio
embeddings_tst_tensors=[]
for inp_str in tst_audio_files:
    inp, sr = torchaudio.load(inp_str)
    
    try:
        embeddings= s2vec_model.predict([inp])
    except:
        embeddings=torch.randn(1,1024)
        print(inp_str)
        print(inp.shape)
    
    embeddings_tst_tensors.append(embeddings)
embeddings_tst= torch.cat(embeddings_tst_tensors, dim=0)
print(embeddings_tst.shape)

from SONAR.sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                           tokenizer="text_sonar_basic_encoder")
sentences_term = term_lines
embeddings_term = t2vec_model.predict(sentences_term, source_lang="eng_Latn")
print(embeddings_term.shape)

# Normalize the vectors (dim=1 is the vector dimension)
normalized_tst = embeddings_tst / embeddings_tst.norm(dim=1, keepdim=True)
normalized_term = embeddings_term / embeddings_term.norm(dim=1, keepdim=True)

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

with open('/home/sli/DPR_SONAR/results/tst_term_pairs_dpr_sonar_q_n_p_audio_text_10.txt','w') as f:
        for item in tst_term_pairs:
               f.write('{} \n'.format(item))

for i,pair in enumerate(tst_term_pairs):
    if i==0:
        with open('/home/sli/DPR_SONAR/results/tst_with_retrieved_example_dpr_sonar_q_n_p_audio_text_1st_NN_3.txt','w') as g: #
            g.write(term_lines[pair[1][0]].strip()+' <SEP> '+lines_tst_origin[pair[0]])
        with open('/home/sli/DPR_SONAR/results/tst_with_retrieved_example_dpr_sonar_q_n_p_audio_text_2st_NN_3.txt','w') as g: #
            g.write(term_lines[pair[1][1]].strip()+' <SEP> '+lines_tst_origin[pair[0]])
        with open('/home/sli/DPR_SONAR/results/tst_with_retrieved_example_dpr_sonar_q_n_p_audio_text_3st_NN_3.txt','w') as g: #
            g.write(term_lines[pair[1][2]].strip()+' <SEP> '+lines_tst_origin[pair[0]])
    else:
        with open('/home/sli/DPR_SONAR/results/tst_with_retrieved_example_dpr_sonar_q_n_p_audio_text_1st_NN_3.txt','a') as g: #
            g.write(term_lines[pair[1][0]].strip()+' <SEP> '+lines_tst_origin[pair[0]])
        with open('/home/sli/DPR_SONAR/results/tst_with_retrieved_example_dpr_sonar_q_n_p_audio_text_2st_NN_3.txt','a') as g: #
            g.write(term_lines[pair[1][1]].strip()+' <SEP> '+lines_tst_origin[pair[0]])
        with open('/home/sli/DPR_SONAR/results/tst_with_retrieved_example_dpr_sonar_q_n_p_audio_text_3st_NN_3.txt','a') as g: #
            g.write(term_lines[pair[1][2]].strip()+' <SEP> '+lines_tst_origin[pair[0]])
