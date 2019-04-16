#!/bin/bash

#WARNING: Need to create a Spark cluster first (dataproc_preprocessing/create_cluster.sh) on GCP Dataproc, 
#open a Jupyter session (dataproc_preprocessing/browse_cluster.sh),
#upload and run the first preprocessing notebook (dataproc_preprocessing/nar_preprocessing_addressa_01_dataproc.ipynb),
#and download the exported sessions JSON lines and the pickle with nar_encoders_dict to be used by this 2nd step of pre-processing


DATA_DIR="[REPLACE WITH THE PATH TO UNZIPPED ADRESSA DATASET FOLDER]" && \

#Testing pre-processing with new contextual features
python3 -m nar.preprocessing.nar_preprocess_adressa \
	--input_sessions_json_folder_path ${DATA_DIR}/sessions_processed_by_spark \
	--input_acr_metadata_embeddings_path ${DATA_DIR}/pickles/acr_articles_metadata_embeddings.pickle \
	--input_nar_encoders_dict_path ${DATA_DIR}/pickles/nar_encoders_adressa.pickle \
	--number_hours_to_preprocess 384 \
 	--output_nar_preprocessing_resources_path ${DATA_DIR}/pickles/nar_preprocessing_resources.pickle \
 	--output_sessions_tfrecords_path "${DATA_DIR}/sessions_tfrecords_by_hour/adressa_sessions_hour_*.tfrecord.gz"