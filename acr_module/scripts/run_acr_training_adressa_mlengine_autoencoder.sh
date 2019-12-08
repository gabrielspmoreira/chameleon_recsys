#!/bin/bash

PROJECT_ID=[REPLACE BY THE GCP PROJECT ID. e.g. 'chameleon-test'] && \
DATA_DIR=[REPLACE BY THE GCS PATH OF ADRESSA ARTICLES DATASET e.g. gs://chameleon_datasets/adressa] && \
JOB_PREFIX=adressa_acr && \
JOB_ID=`whoami`_${JOB_PREFIX}_`date '+%Y_%m_%d_%H%M%S'` && \
MODEL_DIR="[REPLACE BY THE GCS PATH TO OUTPUT ACR MODEL RESULTS. e.g. gs://chameleon_jobs/adressa/acr_module/${JOB_ID}]" && \
JOBS_STAGING_DIR="[REPLACE BY A GCS PATH FOR STAGING. e.g. gs://mlengine_staging/" && \
echo 'Running training job and outputing to '${MODEL_DIR} && \
gcloud --project ${PROJECT_ID} ml-engine jobs submit training ${JOB_ID} \
	--package-path acr \
	--module-name acr.acr_trainer_adressa \
	--staging-bucket ${JOBS_STAGING_DIR} \
	--region us-central1 \
	--python-version 3.5 \
	--runtime-version 1.12 \
	--scale-tier basic-gpu \
	--job-dir ${MODEL_DIR} \
	-- \
	--model_dir ${MODEL_DIR} \
	--train_set_path_regex "${DATA_DIR}/articles_tfrecords/adressa_articles_*.tfrecord.gz" \
	--input_word_vocab_embeddings_path ${DATA_DIR}/pickles/acr_word_vocab_embeddings.pickle \
	--input_label_encoders_path ${DATA_DIR}/pickles/acr_label_encoders.pickle \
	--output_acr_metadata_embeddings_path ${DATA_DIR}/pickles/acr_articles_metadata_embeddings_unsupervised_gru_${JOB_ID}.pickle \
	--batch_size 128 \
	--truncate_tokens_length 30 \
	--training_epochs 5 \
	--learning_rate 3e-4 \
	--dropout_keep_prob 1.0 \
	--l2_reg_lambda 1e-5 \
	--text_feature_extractor "GRU" \
	--training_task "autoencoder" \
	--autoencoder_noise 0.0 \
	--cnn_filter_sizes "3,4,5" \
	--cnn_num_filters 128 \
	--rnn_units 512 \
	--rnn_layers 1 \
	--rnn_direction "unidirectional" \
	--acr_embeddings_size 250


#--input_word_vocab_embeddings_path ${DATA_DIR}/data_transformed/pickles/acr_word_vocab_embeddings.pickle \
#	--input_label_encoders_path ${DATA_DIR}/data_transformed/pickles/acr_label_encoders.pickle \
	