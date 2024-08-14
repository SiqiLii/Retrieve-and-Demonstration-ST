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
python preprocessing/datasets_generation.py --dir ${MUSTC_ROOT}
```

### Data Splits with Gold Example Generation
Construct datasets that prepend gold example to rare-word-tst split, rare-word-dev split, tst-COMMON split respectively, to form tst_ex split, dev_ex split, tst-COMMON_ex
```bash
python preprocessing/datasets_gold_example_generation.py --dir ${MUSTC_ROOT}
```
Construct train_ex split that each sentence in the reduced-train split is prepended sentences that contains the same sentence-level rare word from the reduced-train split
```bash
python preprocessing/datasets_gold_example_generation.py --dir ${MUSTC_ROOT} --train True
```
Download our generated datasets if you want to use our pre-trained models:
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
#### Training
En-De as example:
```bash
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_asr.yaml --train-subset train_ex_asr --valid-subset dev_ex_asr \
  --save-dir ${ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```
where `ASR_SAVE_DIR` is the checkpoint root path.

#### Inference 
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_asr.yaml --gen-subset tst-COMMON_ex_asr --task speech_to_text \
  --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct
```

#### Translation Performance and Accuracy Evaluation
```bash
python postprocessing/analyze.py
```
#### Results
| Data | --arch  | WER(tst-COMMON) | WER(tst) | Translation Accuracy(tst) | Model |
|---|---|---|---|---|---|
| en-de | s2t_transformer_s | 19.1 | 18.1 | 17.7 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|


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
where `ASR_SAVE_DIR` is the existing ASR model trained without example provided in fairseq. [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_de_asr_transformer_s.pt)


#### Adapted ST-Training
En-De as example:
```bash
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
  --save-dir ${Adapted_ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --load-pretrained-encoder-from ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}
```

#### Inference 
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --gen-subset tst-COMMON_st --task speech_to_text \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu
```

```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${Adapted_ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${Adapted_ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --gen-subset tst-COMMON_ex_st --task speech_to_text \
  --path ${Adapted_ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu
```

#### Translation Performance and Accuracy Evaluation
```bash
python postprocessing/analyze.py
```
#### Results
| Data | --arch  | BLEU(tst-COMMON) | BLEU(tst) | Translation Accuracy(tst) | Model |
|---|---|---|---|---|---|
| en-de | s2t_transformer_s | 19.1 | 18.1 | 17.7 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|
| en-de | s2t_transformer_s(adapted) | 19.1 | 18.1 | 17.7 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|

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
python DPR_SONAR/data_formatting.py 
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

### Retriever training
Retriever training quality depends on its effective batch size. We are setting batch size equals to 4.
In order to start training on one machine:

```bash
python train_dense_encoder.py \
train_datasets=[nq_train] \
dev_datasets=[nq_dev] \
train=biencoder_local \
question=text \
ctx=text \
output_dir={path to checkpoints dir}
```
question and ctx referece modalities of question and passages are speech or text.
### Retriever Inference:
We are evaluate retriever on retrieving example sentences from rare-word pool split for sentences from the rare-word tst split.

```bash
python retrieve.py \
	model_file={path to a checkpoint downloaded from our download_data.py as 'checkpoint.retriever.single.nq.bert-base-encoder'} \
	qa_dataset={the name os the test source} \
	ctx_datatsets=[{list of passage sources's names, comma separated without spaces}] \
	encoded_ctx_files=[{list of encoded document files glob expression, comma separated without spaces}] \
	out_file={path to output json file with results} 
	
```
#### Results
| Data | --modality  | retrieve accuracy(tst) | Model |
|---|---|---|---|
| en-de | text-text | 19.1 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|
| en-de | speech-text | 19.1 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|
| en-de | speech-speech | 19.1 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt)|



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
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${Adapted_ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${Adapted_ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --gen-subset tst-COMMON_ex_st --task speech_to_text \
  --path ${Adapted_ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu
```

#### Results
| Data | --modality  | BLEU(tst-COMMON) | BLEU(tst) | Translation Accuracy(tst) |
|---|---|---|---|---|
| en-de | text-to-text | 19.1 | 18.1 | 17.7 |
| en-de | speech-to-text | 19.1 | 18.1 | 17.7 |
| en-de | speech-to-speech | 19.1 | 18.1 | 17.7 |