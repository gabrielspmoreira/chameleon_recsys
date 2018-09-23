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
	--config nar_mlengine_hypertuning.yaml  \
	--job-dir ${MODEL_DIR} \
	-- \
	--model_dir ${MODEL_DIR} \
	--use_local_cache_model_dir \
	--train_set_path_regex "${DATA_DIR}/data_preprocessed_open_dlrs/sessions_tfrecords/sessions_hour_*.tfrecord.gz" \
	--train_files_from 0 \
	--train_files_up_to 48 \
	--training_hours_for_each_eval 5 \
	--save_results_each_n_evals 3 \
	--acr_module_articles_metadata_csv_path ${DATA_DIR}/articles_metadata.csv \
	--acr_module_articles_content_embeddings_pickle_path ${DATA_DIR}/articles_embeddings.pickle \
	--truncate_session_length 20 \
	--recent_clicks_buffer_size 2000 \
	--eval_metrics_top_n 5 \
	--rnn_num_layers 1 \
	--train_negative_samples_from_buffer 10 \
	--eval_total_negative_samples 50 \
	--eval_negative_samples_from_buffer 50 \
	--disable_eval_benchmarks