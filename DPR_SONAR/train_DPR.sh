nohup python train_dense_encoder_DPR.py \
    train_datasets=["/home/sli/DPR/downloads/data/fairseq/data_train_audio_new_2.json"] \
    dev_datasets=["/home/sli/DPR/downloads/data/fairseq/data_dev_audio.json"] \
    train=biencoder_default \
    model_file="/export/data2/sli/data/DPR/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp" \
    output_dir="/home/sli/DPR_t5/results_text_text_DPR/" > train_text_text_DPR.log &