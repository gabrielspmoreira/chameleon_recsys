#!/bin/bash
DATA_DIR="[REPLACE BY THE G1 ARTICLES DATASET PATH]" && \
python3 -m acr.preprocessing.acr_preprocess_gcom \
	--input_articles_csv_path ${DATA_DIR}/document_g1/documents_g1.csv \
 	--input_word_embeddings_path ${DATA_DIR}/word2vec/skip_s300.txt \
 	--vocab_most_freq_words 50000 \
 	--output_word_vocab_embeddings_path ${DATA_DIR}/pickles/acr_word_vocab_embeddings.pickle \
 	--output_label_encoders ${DATA_DIR}/pickles/acr_label_encoders.pickle \
 	--output_tf_records_path "${DATA_DIR}/articles_tfrecords/gcom_articles_tokenized_*.tfrecord.gz" \
 	--articles_by_tfrecord 5000


