#!/bin/bash

DATA_DIR="[REPLACE WITH THE PATH TO UNZIPPED GCOM DATASET FOLDER]" && \
time python3 -m  nar.benchmarks.gru4rec.run_gru4rec \
--dataset_type "gcom" \
--train_set_path_regex "${DATA_DIR}/sessions_tfrecords/sessions_hour_*.tfrecord.gz" \
--eval_sessions_negative_samples_json_path "${DATA_DIR}/eval_sessions_negative_samples.json" \
--acr_module_resources_path ${DATA_DIR}/data_preprocessed/pickles_v3/acr_articles_metadata_embeddings.pickle \
--training_hours_for_each_eval 5 \
--eval_metrics_top_n 10 \
--batch_size 128 \
--n_epochs 3 \
--optimizer "adam" \
--dropout_p_hidden 0.0 \
--learning_rate 1e-4 \
--l2_lambda 1e-5 \
--momentum 0.0 \
--embedding 0 \
--recent_clicks_buffer_hours 1.0 \
--recent_clicks_buffer_max_size 20000 \
--recent_clicks_for_normalization 2000 \
--eval_negative_sample_relevance 0.02