# Retrieve-and-Demonstration-ST
Enhancing Rare Word Translation Accuracy in direct Speech Translation via Retrieve-and-Demonstration approach

## Dataset Construction

We are using english-to-german translation language pair of the [MuST-C](https://www.aclweb.org/anthology/N19-1202) dataset(v.2) for training and evaluation, which is a multilingual speech-to-text translation corpus on English TED talks.

[Download](https://ict.fbk.eu/must-c) (en-de language pair) and unpack MuST-C data to a path
`${MUSTC_ROOT}/en-de`

### Rare Word List Generation
We first take the train-split to evaluate how many rare words are there and where do they appear, write them to a rare word list at rareword_terminology.txt. Here's our generated [rare word list](https://ict.fbk.eu/rareword_terminology.txt)
```bash
python preprocessing/rareword_analyze.py ${MUSTC_ROOT}/train/txt/train.en
```
Then align rare words with their german translation.
First install [Awesomealign](https://github.com/neulab/awesome-align) 
```bash
python preprocessing/awesomealign_data_generation.py ${MUSTC_ROOT}/train/txt/train.en  ${MUSTC_ROOT}/train/txt/train.de
cd awesome-align

DATA_FILE=train_ende.src-tgt
MODEL_NAME_OR_PATH=bert-base-multilingual-cased
OUTPUT_FILE=train_alignment.txt
OUTPUT_WORD_FILE=train_alignment_word.txt
    awesome-align \
    --output_file=$OUTPUT_FILE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --data_file=$DATA_FILE \
    --extraction 'softmax' \
    --batch_size 32 \
    --output_word_file=$OUTPUT_WORD_FILE
```
```bash
python preprocessing/rareword_alignment.py /path/to/rareword_terminology.txt /path/to/train_alignment_word.txt
```
Here we align rare english word with its german translation, at rareword_terminology_de.txt. Here's our generated rare word [translation](https://ict.fbk.eu/rareword_terminology_de.txt)
### Data Splits Generation
We are using analyzed rare word information to construct reduced-train split, rare-word-tst split, rare-word-dev split, rare-word-pool: named train, tst, dev, terminology respectively
```bash
python preprocessing/datasets_generation.py ${MUSTC_ROOT} /path/to/rareword_terminology /path/to/rareword_terminology_de /path/to/train_en

python preprocessing/datasets_no_example_generation.py 
    --path_origin_yaml $YAML_FILE_PATH_TO_ORIGINAL_TRAIN_SET \
    --path_origin_en $EN_FILE_PATH_TO_ORIGINAL_TRAIN_SET \
    --path_origin_de  $DE_FILE_PATH_TO_ORIGINAL_TRAIN_SET \
    --root_path_origin $DIR_PATH_TO_ORIGINAL_TRAIN_SET \
    --new_root_path  \
    --name \
    --file_path_index_list  \
```

### Data Splits with Gold Example Generation
Construct datasets that prepend gold example to rare-word-tst split, rare-word-dev split, tst-COMMON split respectively, to form tst_ex split, dev_ex split, tst-COMMON_ex
```bash
python preprocessing/datasets_gold_example_generation.py 
    --root_path_example \
    --path_example_yaml \
    --path_example_en  \
    --path_example_de  \
    --path_origin_yaml \
    --path_origin_en \
    --path_origin_de  \
    --root_path_origin  \
    --new_root_path  \
    --name \
    --file_path_index_pair  \
```
Construct train_ex split that each sentence in the reduced-train split is prepended sentences that contains the same sentence-level rare word from the reduced-train split
```bash
python preprocessing/datasets_gold_example_generation.py --dir ${MUSTC_ROOT} --train True
```
Download our generated datasets:
- Reduced-Datasets: [Train](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link), [dev](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link), [rare-word-tst](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link),[rare-word-pool](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link)
- Reduced-Datasets with gold example: [Train_ex](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link), [dev_ex](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link), [tst_ex](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link),[tst-COMMON_ex](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link)


## ASR and ST training

### Installation

Installation from the source. Python's virtual or Conda environments are recommended.

```bash
cd fairseq-adapted
pip install .
```
### Preprocessing

```bash
# additional Python packages for S2T data processing/model training
pip install pandas torchaudio soundfile sentencepiece

# Generate TSV manifests, features, vocabulary
# and configuration for each language
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task asr \
  --vocab-type unigram --vocab-size 5000
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st \
  --vocab-type unigram --vocab-size 8000

```
The generated files (manifest, features, vocabulary and data configuration) will be added to
`${MUSTC_ROOT}/en-de` (per-language data) and `MUSTC_ROOT` (joint data).

Download our vocabulary files if you want to use our pre-trained models:
- ASR: [En-De](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_de_asr_vocab_unigram5000.zip)
- ST: [En-De](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_de_st_vocab_unigram8000.zip)

### ASR
#### ASR Training
En-De as example:
```bash
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
  --save-dir ${ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```
where `ASR_SAVE_DIR` is the checkpoint root path.

#### Adapted ASR Training
En-De as example:
```bash
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_asr.yaml --train-subset train_ex_asr --valid-subset dev_ex_asr \
  --save-dir ${Adapted_ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --arch s2t_transformer_s --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt --ignore-prefix-size 1 \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```
where `Adapted_ASR_SAVE_DIR` is the checkpoint root path of Adapted ASR checkpoint.

#### ASR Inference 
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_asr.yaml --gen-subset tst_asr --task speech_to_text \
  --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct
```

#### Adapted ASR Inference 
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_asr.yaml --gen-subset tst_ex_asr --task speech_to_text \
  --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct --batch-size 1 --prefix-size 1 --max-source-positions 10000
```

#### Translation Performance and Accuracy Evaluation
```bash
python postprocessing/analyze.py path/to/inference/log path/to/rareword_terminology path/to/rareword_terminology_de path/to/coorsponding_english_transcript /path/to/analyzed_file_name analyze_type/ASR_or_ST
```
#### Results
| Data | --arch  | WER(tst) | Translation Accuracy(tst) | Model |
|---|---|---|---|---|
| en-de | s2t_transformer_s | 14.8 | 31.2% | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|
| en-de | s2t_transformer_s(adapted) | 22.0 | 72.1% | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|


### ST
#### ST-Training
En-De as example:
```bash
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
  --save-dir ${ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}
```
where `ASR_SAVE_DIR` is the ASR model trained without example as given in [fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md).


#### Adapted ST-Training
En-De as example:
```bash
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --train-subset train_ex_st --valid-subset dev_ex_st \
  --save-dir ${Adapted_ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.2 \
  --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt  --ignore-prefix-size 1  \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --load-pretrained-encoder-from ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --skip-invalid-size-inputs-valid-test  --finetune-from-model ${ST_SAVE_DIR}/${ST_CHECKPOINT_FILENAME}
```

#### Inference 
```bash
outname=/path/to/inderence_log
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

for p in tst-COMMON_st tst_st; do
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --gen-subset $p --task speech_to_text \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu --max-source-positions 100000 > $outname

hyp=$outname.hyp
ref=$outname.ref
grep "^D-" $outname | sed -e "s/^D-//g"| sort -n | cut -f3- > $outname.hyp
grep "^T-" $outname | sed -e "s/^T-//g"| sort -n | cut -f2- > $outname.ref

p_orig=${p%"_st"}
src=${MUSTC_ROOT}/en-de/data/$p_orig/txt/$p_orig.en

comet-score -s $src -t $hyp -r $ref > $outname.comet
wc -l $hyp $ref
cat $hyp | sacrebleu $ref -m bleu chrf > $outname.bleu 
```

```bash
outname=/path/to/inderence_log
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${Adapted_ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${Adapted_ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

for p in tst-COMMON_ex_st tst_ex_new_st; do
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --gen-subset $p --task speech_to_text \
  --path ${Adapted_ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu --batch-size 1 --prefix-size 1 --max-source-positions 100000 > $outname

hyp=$outname.hyp
ref=$outname.ref
grep "^D-" $outname | sed -e "s/^D-//g"| sort -n | cut -f3- > $outname.hyp_pre
grep "^T-" $outname | sed -e "s/^T-//g"| sort -n | cut -f2- > $outname.ref_pre
awk '/<SEP>/ {print substr($0, index($0, "<SEP>") + 6)} !/<SEP>/ {print ""}' $outname.hyp_pre > $outname.hyp
awk '/<SEP>/ {print substr($0, index($0, "<SEP>") + 6)} !/<SEP>/ {print ""}' $outname.ref_pre > $outname.ref

p_orig=${p%"_st"}
src=${MUSTC_ROOT}/en-de/data/$p_orig/txt/$p_orig.en

comet-score -s $src -t $hyp -r $ref > $outname.comet
wc -l $hyp $ref
cat $hyp | sacrebleu $ref -m bleu chrf > $outname.bleu 
```

#### Translation Performance and Accuracy Evaluation
```bash
python postprocessing/analyze.py path/to/inference/log path/to/rareword_terminology path/to/rareword_terminology_de path/to/coorsponding_english_transcript /path/to/analyzed_file_name analyze_type/ASR_or_ST
```
#### Results
| Data | --arch  | BLEU(tst-COMMON) | COMET(tst-COMMON) | BLEU(tst) | COMET(tst) | Translation Accuracy(tst) | Model |
|---|---|---|---|---|---|---|---|
| en-de | s2t_transformer_s | 23.6 | 70.5 | 17.2 | 57.9 | 11.8% | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|
| en-de | s2t_transformer_s(adapted) | 21.8 | 64.5 | 17.0 | 55.6 | 29.4% | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|

## Retriever Training

### Installation

Installation from the source. Python's virtual or Conda environments are recommended.

```bash
cd DPR_SONAR
pip install .
```
### Data formats
First, you need to prepare data for our retriever training.

```bash
python DPR_SONAR/data_formatting.py /path/to/rareword_terminology /path/to/train_en /path/to/dev_ex_en_path
```
The default data format of the Retriever training data is JSON.
It contains pools of negative passages, positive passages per question, and some additional information. Positive passages are sentences containing the same rare word, and negative passages are sentences not sharing rare words.

```
[
  {
	"question": "....",
	"answers": ["...", "...", "..."],
	"positive_ctxs": [{
		"title": "...",
		"text": "...."
	}],
	"negative_ctxs": ["..."],
	"hard_negative_ctxs": ["..."]
  },
  ...
]
```
Our generated training data [data_train_audio_new_2.json](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt) and dev data [data_dev_audio.json](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt) is available here.

### Retriever training
Retriever training quality depends on its effective batch size. We are setting batch size equals to 4.
In order to start training on one machine:
edit init_encoder function of DPR_SONAR/src/dpr_sonar/models/sonar_model.py for freeze different number of layers of SONAR encoder
```bash
cd DPR_SONAR
pip install .

python train_dense_encoder_SONAR.py \
train_datasets=["/path/to/data_train_audio_new_2.json"] \
dev_datasets=["/path/to/data_dev_audio.json"] \
train=biencoder_default \
output_dir=${path to checkpoints dir}
```
Set 'question_input_type' and 'ctx_input_type' as 'audio' or 'text' in DPR_SONAR/conf/biencoder_train_cfg_SONAR.yaml for retrieval modalities of question and passages are speech or text.
speech-to-speech retrieval:question_input_type:'audio', ctx_input_type:'audio'
speech-to-text retrieval:question_input_type:'audio', ctx_input_type:'text'
text-to-text retrieval:question_input_type:'text', ctx_input_type:'text'
### Retriever Inference:
We are evaluate retriever on retrieving example sentences from rare-word pool split for sentences from the rare-word tst split.

```bash
python DPR_SONAR/retrieve.py  /path/to/tst_en /path/to/train_en /path/to/term_en /path/to/term_wav_dir /path/to/tst_wav_dir /path/to/retreiver_model_checkpoint /path/to/rareword_terminology query_type ctx_type 
	
```
#### Results
| Data | --modality  | retrieve accuracy(tst) | Model |
|---|---|---|---|
| en-de | text-text | 46.6 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|
| en-de | speech-text | 33.3 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|
| en-de | speech-speech | 41.3 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|



## Evaluate on Retrieved Data

### Prepend retrieved example to tst-split
```bash
python data_generation_retriever.py \
	model_file={path to a checkpoint downloaded from our download_data.py as 'checkpoint.retriever.single.nq.bert-base-encoder'} \
	qa_dataset={the name os the test source} \
	ctx_datatsets=[{list of passage sources's names, comma separated without spaces}] \
	encoded_ctx_files=[{list of encoded document files glob expression, comma separated without spaces}] \
	out_file={path to output json file with results} 
```
### Data Preparation
```bash
# Generate TSV manifests, features, vocabulary
# and configuration for each language
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task asr \
  --vocab-type unigram --vocab-size 5000
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st \
  --vocab-type unigram --vocab-size 8000

```
generate tsv files for inference

### Inference and Evaluation
Inference with retrieved example prepended tst split
```bash
outname=/path/to/inderence_log
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${Adapted_ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${Adapted_ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

for p in tst_audio_audio_st tst_audio_text_st tst_text_text_st; do
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --gen-subset $p --task speech_to_text \
  --path ${Adapted_ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu --batch-size 1 --prefix-size 1 --max-source-positions 100000 > $outname

hyp=$outname.hyp
ref=$outname.ref
grep "^D-" $outname | sed -e "s/^D-//g"| sort -n | cut -f3- > $outname.hyp_pre
grep "^T-" $outname | sed -e "s/^T-//g"| sort -n | cut -f2- > $outname.ref_pre
awk '/<SEP>/ {print substr($0, index($0, "<SEP>") + 6)} !/<SEP>/ {print ""}' $outname.hyp_pre > $outname.hyp
awk '/<SEP>/ {print substr($0, index($0, "<SEP>") + 6)} !/<SEP>/ {print ""}' $outname.ref_pre > $outname.ref

p_orig=${p%"_st"}
src=${MUSTC_ROOT}/en-de/data/$p_orig/txt/$p_orig.en

comet-score -s $src -t $hyp -r $ref > $outname.comet
wc -l $hyp $ref
cat $hyp | sacrebleu $ref -m bleu chrf > $outname.bleu 
```

#### Results
| Data | --modality  | BLEU(tst-COMMON) | COMET(tst-COMMON) | BLEU(tst) | COMET(tst) | Translation Accuracy(tst) |
|---|---|---|---|---|---|---|
| en-de | text-to-text | 21.62 | 64 | 15.2 | 54.4 | 20.1%
| en-de | speech-to-text | 21.35 | 64.1 | 15.3 | 54.0 | 18.8%
| en-de | speech-to-speech | 21.82 | 64.9 | 16.2 | 55.3 | 20.3%