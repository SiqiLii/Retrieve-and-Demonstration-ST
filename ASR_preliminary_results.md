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
