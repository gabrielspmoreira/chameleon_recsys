#!/bin/bash

DATA_DIR="[REPLACE WITH THE PATH TO UNZIPPED GCOM DATASET FOLDER]" && \
time python3 -m  nar.benchmarks.gru4rec.run_gru4rec \
--train_set_path_regex "${DATA_DIR}/sessions_tfrecords/sessions_hour_*.tfrecord.gz" \
--eval_sessions_negative_samples_json_path "${DATA_DIR}/eval_sessions_negative_samples.json" \
--training_hours_for_each_eval 5 \
--eval_metrics_top_n 5 \
--batch_size 128 \
--n_epochs 1 \
--optimizer "adam" \
--dropout_p_hidden 0.0 \
--learning_rate 1e-4 \
--l2_lambda 1e-5 \
--momentum 0.0 \
--embedding 0