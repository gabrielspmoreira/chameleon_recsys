#!/bin/bash

DATA_DIR="[REPLACE WITH THE PATH TO UNZIPPED ADRESSA DATASET FOLDER]" && \
time python3 -m  nar.benchmarks.sr-gnn.run_sr_gnn \
--dataset_type "adressa" \
--train_set_path_regex "${DATA_DIR}/sessions_tfrecords/sessions_hour_*.tfrecord.gz" \
--eval_sessions_negative_samples_json_path "${DATA_DIR}/eval_sessions_negative_samples.json" \
--acr_module_resources_path ${DATA_DIR}/data_transformed/pickles/acr_articles_metadata_embeddings_v2_body_included_30epochs.pickle \
--training_hours_for_each_eval 5 \
--eval_metrics_top_n 10 \
--batch_size 128 \
--n_epochs 10 \
--hidden_size 200 \
--l2_lambda 1e-5 \
--propagation_steps 1 \
--learning_rate 0.001 \
--learning_rate_decay 0.1 \
--learning_rate_decay_steps 3 \
--nonhybrid \
--recent_clicks_buffer_hours 1.0 \
--recent_clicks_buffer_max_size 30000 \
--recent_clicks_for_normalization 5000 \
--eval_negative_sample_relevance 0.02
