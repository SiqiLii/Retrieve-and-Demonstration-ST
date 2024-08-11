nohup python train_dense_encoder.py \
    train_datasets=["/home/sli/DPR/downloads/data/fairseq/data_train_audio_new_2.json"] \
    dev_datasets=["/home/sli/DPR/downloads/data/fairseq/data_dev_audio.json"] \
    train=biencoder_default \
    output_dir="/home/sli/DPR_t5/results_text_text" > train_text_text.log &