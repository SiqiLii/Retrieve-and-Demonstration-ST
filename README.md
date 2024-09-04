# Retrieve-and-Demonstration-ST

This repo contains code for the paper 	
[Optimizing Terminology Accuracy in Speech Translation with a Retrieval-and-Demonstration Approach](TODO:add arxiv link).

## Setup

```bash
# create and activate env
conda create -n rareword-ST python=3.9
conda activate rareword-ST
# pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge 
```

For the speech translation module: 
```bash
# use earlier version omegaconf (required by fairseq)
pip install omegaconf==2.0.6
# install fairseq
cd fairseq-adapted
pip install .
```

For the retriver:
```bash
cd DPR_SONAR
pip install .
```

## Data Preparation

For details on the dataset splitting, please see [here](preprocessing/README.md). 

Download our generated datasets:
- Reduced-Datasets: [Train](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link), [dev](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link), [rare-word-tst](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link),[rare-word-pool](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link)
- Reduced-Datasets with gold example: [Train_ex](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link), [dev_ex](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link), [tst_ex](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link),[tst-COMMON_ex](https://drive.google.com/drive/folders/1mrR67koMGwbtrmyyzaGhUBzQvucARUFB?usp=drive_link)

## Speech Translation

### Preprocessing

In general, we follow the MuST-C preprocessing steps from [fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md#data-preparation):
```bash
python $FAIRSEQ_DIR/examples/speech_to_text/prep_mustc_data.py \
        --data-root ${MUSTC_ROOT} --task st \
        --vocab-type unigram --vocab-size 8000
```

To prevent the tokenizer from seeing the rare words during its training,
we create a different vocabulary from the fairseq example on the reduced train set after the utterances containing rare words are moved to dedicated splits.
- [vocabulary files](TODO)

## Training

### Baseline

For training the baseline ST model, 
we follow the setup from [fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md#training-1):

<details>
<summary><b> Training script </b></summary>

```bash
DATADIR=#path to data

arch=s2t_transformer_s
SAVEDIR=/export/data2/dliu/models/mustc_st_en_de_$arch
mkdir -p $SAVEDIR

# pretrained ASR checkpoint from https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_de_asr_transformer_s.pt
asr_ckpt=/export/data2/dliu/models/mustc_asr_en_de_fairseq/mustc_de_asr_transformer_s.pt

CUDA_VISIBLE_DEVICES=3 fairseq-train $DATADIR \
  --config-yaml config_st.yaml --train-subset train_st  --valid-subset dev_st \
  --save-dir $SAVEDIR --num-workers 4 --max-tokens 80000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --arch $arch --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --load-pretrained-encoder-from $asr_ckpt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 4 --keep-last-epochs 1 --no-epoch-checkpoints \
  --save-interval-updates 1000 --keep-interval-updates 10 \
  --max-epoch 150 --skip-invalid-size-inputs-valid-test --patience 30 2>&1 | tee -a $SAVEDIR/train.log
```
</details>


### Adapted ST-Training

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

### Inference

<details>
<summary><b> Inference script </b></summary>

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

</details>

<details>
<summary><b> Inference script (without prepended examples) </b></summary>

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

</details>

### Evaluation

```bash
python postprocessing/analyze.py path/to/inference/log path/to/rareword_terminology path/to/rareword_terminology_de path/to/coorsponding_english_transcript /path/to/analyzed_file_name analyze_type/ASR_or_ST
```
#### Results
| Data | --arch  | BLEU(tst-COMMON) | COMET(tst-COMMON) | BLEU(tst) | COMET(tst) | Translation Accuracy(tst) | Model |
|---|---|---|---|---|---|---|---|
| en-de | s2t_transformer_s | 23.6 | 70.5 | 17.2 | 57.9 | 11.8% | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|
| en-de | s2t_transformer_s(adapted) | 21.8 | 64.5 | 17.0 | 55.6 | 29.4% | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|


## Retriever 

### Data formats
First, prepare data for our retriever training.

```bash
python DPR_SONAR/data_formatting.py /path/to/rareword_terminology /path/to/train_en /path/to/dev_ex_en_path
```
The default data format of the Retriever training data is JSON.
It contains pools of negative passages, positive passages per question, and some additional information. 
Positive passages are sentences containing the same rare word, and negative passages are sentences not sharing rare words.

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
Our generated training data [data_train_audio_new_2.json](https://drive.google.com/file/d/1AQ_9DoDjjEHjyEM1f7-ZSGA6nOjq919i/view?usp=drive_link) and dev data [data_dev_audio.json](https://drive.google.com/file/d/10W6CDXdGg787mwaIlzUR4ZQgniYOpMgK/view?usp=drive_link) is available here.

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