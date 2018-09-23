#!/bin/bash

PROJECT_ID="[REPLACE BY THE GCP PROJECT ID. e.g. 'chameleon-test']" && \
DATA_DIR="[REPLACE BY THE GCS PATH OF GCOM DATASET. e.g. gs://chameleon_datasets/gcom]" && \
JOB_PREFIX=gcom_nar && \
JOB_ID=`whoami`_${JOB_PREFIX}_`date '+%Y_%m_%d_%H%M%S'` && \
MODEL_DIR="[REPLACE BY THE GCS PATH TO OUTPUT NAR MODEL RESULTS. e.g. gs://chameleon_jobs/gcom/nar_module/${JOB_ID}]" && \
JOBS_STAGING_DIR="[REPLACE BY A GCS PATH FOR STAGING. e.g. gs://mlengine_staging/" && \
echo 'Running training job '${JOB_ID} && \
gcloud --project ${PROJECT_ID} ml-engine jobs submit training ${JOB_ID} \
	--package-path nar \
	--module-name nar.nar_trainer_gcom_dlrs \
	--staging-bucket ${JOBS_STAGING_DIR} \
	--region us-central1 \
	--python-version 3.5 \
	--runtime-version 1.8 \
	--scale-tier basic-gpu \
	--job-dir ${MODEL_DIR} \
	-- \
	--model_dir ${MODEL_DIR} \
	--use_local_cache_model_dir \
	--train_set_path_regex "${DATA_DIR}/sessions_tfrecords/sessions_hour_*.tfrecord.gz" \
	--train_files_from 0 \
	--train_files_up_to 72 \
	--training_hours_for_each_eval 5 \
	--save_results_each_n_evals 3 \
	--acr_module_articles_metadata_csv_path ${DATA_DIR}/articles_metadata.csv \
	--acr_module_articles_content_embeddings_pickle_path ${DATA_DIR}/articles_embeddings.pickle \
	--batch_size 256 \
	--truncate_session_length 20 \
	--learning_rate 0.001 \
	--dropout_keep_prob 1.0 \
	--reg_l2 0.0001 \
	--cosine_loss_gamma 5.0 \
	--recent_clicks_buffer_size 2000 \
	--eval_metrics_top_n 5 \
	--CAR_embedding_size 1024 \
	--rnn_units 255 \
	--rnn_num_layers 1 \
	--train_total_negative_samples 7 \
	--train_negative_samples_from_buffer 10 \
	--eval_total_negative_samples 50 \
	--eval_negative_samples_from_buffer 50 \
	--save_eval_sessions_negative_samples

#To warm start model with previously trained model:
#--warmup_model_dir "gs://chameleon_jobs/gcom/nar_module/gabrielpm_gcom_nar_2018_08_23_230202" \