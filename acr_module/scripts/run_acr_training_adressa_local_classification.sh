#!/bin/bash

DATA_DIR="[REPLACE BY THE ADRESSA ARTICLES DATASET PATH]" && \
JOB_PREFIX=adressa && \
JOB_ID=`whoami`_${JOB_PREFIX}_`date '+%Y_%m_%d_%H%M%S'` && \
MODEL_DIR='/tmp/chameleon/addressa/jobs/'${JOB_ID} && \
echo 'Running training job and outputing to '${MODEL_DIR} && \
python3 -m acr.acr_trainer_adressa \
	--model_dir ${MODEL_DIR} \
	--train_set_path_regex "${DATA_DIR}/articles_tfrecords/adressa_articles_*.tfrecord.gz" \
	--input_word_vocab_embeddings_path ${DATA_DIR}/pickles/acr_word_vocab_embeddings.pickle \
	--input_label_encoders_path ${DATA_DIR}/pickles/acr_label_encoders.pickle \
	--output_acr_metadata_embeddings_path ${DATA_DIR}/pickles/acr_articles_metadata_embeddings_supervised_cnn.pickle \
	--batch_size 8 \
	--truncate_tokens_length 300 \
	--training_epochs 1 \
	--learning_rate 3e-4 \
	--dropout_keep_prob 1.0 \
	--l2_reg_lambda 1e-5 \
	--text_feature_extractor "CNN" \
	--training_task "metadata_classification" \
	--cnn_filter_sizes "3,4,5" \
	--cnn_num_filters 128 \
	--rnn_units 512 \
	--rnn_layers 1 \
	--rnn_direction "unidirectional" \
	--acr_embeddings_size 250