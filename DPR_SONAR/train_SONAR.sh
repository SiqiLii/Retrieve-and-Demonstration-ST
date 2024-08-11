nohup python train_dense_encoder_SONAR.py \
    train_datasets=["/home/sli/DPR/downloads/data/fairseq/data_train_audio_new_2.json"] \
    dev_datasets=["/home/sli/DPR/downloads/data/fairseq/data_dev_audio.json"] \
    train=biencoder_default \
    output_dir="/export/data2/sli/checkpoints/sonar_dpr_finetune_text_text_freeze_20" > train_text_text_SONAR_freeze_20.log &