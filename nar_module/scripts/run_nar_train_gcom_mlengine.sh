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
	--module-name nar.nar_trainer_gcom \
	--staging-bucket ${JOBS_STAGING_DIR} \
	--region us-central1 \
	--python-version 3.5 \
	--runtime-version 1.12 \
	--scale-tier basic-gpu \
	--job-dir ${MODEL_DIR} \
	-- \
	--model_dir ${MODEL_DIR} \
	--use_local_cache_model_dir \
	--train_set_path_regex "${DATA_DIR}/sessions_tfrecords/sessions_hour_*.tfrecord.gz" \
	--train_files_from 0 \
	--train_files_up_to 385 \
	--training_hours_for_each_eval 5 \
	--save_results_each_n_evals 1 \
	--acr_module_articles_metadata_csv_path ${DATA_DIR}/articles_metadata.csv \
	--acr_module_articles_content_embeddings_pickle_path ${DATA_DIR}/articles_embeddings.pickle \
	--batch_size 256 \
	--truncate_session_length 20 \
	--learning_rate 1e-4 \
	--dropout_keep_prob 1.0 \
	--reg_l2 1e-5 \
	--softmax_temperature 0.1 \
	--recent_clicks_buffer_hours 1.0 \
	--recent_clicks_buffer_max_size 20000 \
	--recent_clicks_for_normalization 2000 \
	--eval_metrics_top_n 10 \
	--CAR_embedding_size 1024 \
	--rnn_units 255 \
	--rnn_num_layers 2 \
	--train_total_negative_samples 50 \
	--train_negative_samples_from_buffer 3000 \
	--eval_total_negative_samples 50 \
	--eval_negative_samples_from_buffer 3000 \
	--eval_negative_sample_relevance 0.02 \
	--content_embedding_scale_factor 6.0 \
	--enabled_articles_input_features_groups "category" \
	--enabled_clicks_input_features_groups "time,device,location,referrer" \
	--enabled_internal_features "item_clicked_embeddings,recency,novelty,article_content_embeddings" \
	--novelty_reg_factor 0.0 \
	--disable_eval_benchmarks

#--save_histograms
#--save_eval_sessions_recommendations
#--save_eval_sessions_negative_samples
#--disable_eval_benchmarks
#--eval_cold_start