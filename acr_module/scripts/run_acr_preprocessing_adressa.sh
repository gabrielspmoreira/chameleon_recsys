#!/bin/bash
DATA_DIR="[REPLACE BY THE ADRESSA ARTICLES DATASET PATH]" && \
python3 -m acr.preprocessing.acr_preprocess_adressa \
	--input_articles_folder_path ${DATA_DIR}/data/contentdata \
 	--input_word_embeddings_path ${DATA_DIR}/word_embeddings/w2v_skipgram_no_lemma_aviskorpus_nowac_nbdigital/model.txt \
 	--vocab_most_freq_words 100000 \
	--max_words_length 1000 \
 	--output_word_vocab_embeddings_path ${DATA_DIR}/pickles/acr_word_vocab_embeddings.pickle \
 	--output_label_encoders ${DATA_DIR}/pickles/acr_label_encoders.pickle \
 	--output_tf_records_path "${DATA_DIR}/articles_tfrecords/adressa_articles_*.tfrecord.gz" \
	--output_articles_csv_path "${DATA_DIR}/adressa_articles.csv" \
 	--articles_by_tfrecord 1000