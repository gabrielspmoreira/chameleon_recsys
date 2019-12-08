#!/bin/bash

DATA_DIR="[REPLACE BY THE G1 ARTICLES DATASET PATH]" && \
JOB_PREFIX=gcom && \
JOB_ID=`whoami`_${JOB_PREFIX}_`date '+%Y_%m_%d_%H%M%S'` && \
MODEL_DIR='/tmp/chameleon/gcom/jobs/'${JOB_ID} && \
echo 'Running training job and outputing to '${MODEL_DIR} && \
python3 -m acr.acr_trainer_gcom \
	--model_dir ${MODEL_DIR} \
	--train_set_path_regex "${DATA_DIR}/articles_tfrecords/gcom_articles_tokenized_*.tfrecord.gz" \
	--input_word_vocab_embeddings_path ${DATA_DIR}/pickles/acr_word_vocab_embeddings.pickle \
	--input_label_encoders_path ${DATA_DIR}/pickles/acr_label_encoders.pickle \
	--output_acr_metadata_embeddings_path ${DATA_DIR}/pickles/acr_articles_metadata_embeddings_unsupervised_gru.pickle \
	--batch_size 32 \
	--truncate_tokens_length 30 \
	--training_epochs 20 \
	--learning_rate 1e-4 \
	--dropout_keep_prob 1.0 \
	--l2_reg_lambda 7e-4 \
	--text_feature_extractor "GRU" \
	--training_task "autoencoder" \
	--autoencoder_noise 0.05 \
	--cnn_filter_sizes "3,4,5" \
	--cnn_num_filters 128 \
	--rnn_units 512 \
	--rnn_layers 1 \
	--rnn_direction "unidirectional" \
	--acr_embeddings_size 250