## Dataset Construction

We are using english-to-german translation language pair of the [MuST-C](https://www.aclweb.org/anthology/N19-1202) dataset(v.2) for training and evaluation, which is a multilingual speech-to-text translation corpus on English TED talks.

Download MuST-C v2(en-de language pair) and unpack MuST-C data to a path
`${MUSTC_ROOT}/en-de`.

### Rare Word List Generation
We first take the train-split to evaluate how many rare words are there and where do they appear, write them to a rare word list at rareword_terminology.txt. Here's our generated [rare word list](https://drive.google.com/file/d/1WgL_KKi11o1_j3P8rGdmrD-wcioq4K-X/view?usp=drive_link).
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
Here we align rare english word with its german translation, at rareword_terminology_de.txt. Here's our generated rare word [translation](https://drive.google.com/file/d/1_f7inMAoz_O_O9lq9E_g5XsUWdSFzHr3/view?usp=drive_link)
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