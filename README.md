# CHAMELEON - A Deep Learning Meta-Architecture for News Recommender Systems
CHAMELEON is a Deep Learning Meta-Architecture for News Recommender Systems. It has being developed as part of the Doctoral research of Gabriel de Souza Pereira Moreira, at the Brazilian Aeronautics Institute of Technology ([ITA](http://www.ita.br/)).

The initial version of CHAMELEON source code allows reproducibility of the experiments reported in the following paper (https://arxiv.org/abs/1808.00076) at the [DLRS'18](https://recsys.acm.org/recsys18/dlrs/), co-located with ACM RecSys.   
Please cite as follows:

> Gabriel de Souza Pereira Moreira, Felipe Ferreira, and Adilson Marques da Cunha. 2018. News Session-Based Recommendations using Deep Neural Networks. In 3rd Workshop on Deep Learning for Recommender Systems (DLRS 2018), October 6, 2018, Vancouver, BC, Canada. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3270323.3270328

This implementation depends on **Python 3** (with Pandas, Scikit-learn and SciPy modules) and **TensorFlow 1.8**. CHAMELEON modules were implemented using TF [Estimators](https://www.tensorflow.org/guide/estimators) and [Datasets](https://www.tensorflow.org/guide/datasets).

The CHAMELEON modules training and evaluation can be performed either locally (GPU highly recommended) or using [Google Cloud Platform ML Engine](https://cloud.google.com/ml-engine/) managed service.

# CHAMELEON
The objetive of the CHAMELEON is to provide accurate contextual session-based recommendations for news portals. It is composed of two complementary modules, with independent life cycles for training and inference: the *Article Content Representation (ACR)* and the *Next-Article Recommendation (NAR)* modules, as shown in Figure 1.

![Fig. 1 - CHAMELEON - a Deep Learning Meta-Architecture for News Recommender Systems](https://raw.githubusercontent.com/gabrielspmoreira/chameleon_recsys/master/resources/chameleon-meta.png)
> Fig. 1 - CHAMELEON - a Deep Learning Meta-Architecture for News Recommender Systems

The CHAMELEON is a meta-architecture, in the sense of a reference architecture that collects together decisions relating to an architecture strategy. It might be instantiated as different architectures with similar characteristics that fulfill a common task, in this case, news recommendations.

## Article Content Representation (ACR) module
The *ACR* module is responsible to extract features from news articles text and metadata and to learn a distributed representations (embeddings) for each news article context.

The inputs for the *ACR* module are (1) article metadata attributes (e.g., publisher) and (2) article textual content, represented as a sequence of word embeddings.

In this instantiation of the *Textual Features Representation (TFR)* sub-module from ACR module, 1D CNNs over pre-trained Word2Vec embeddings was used to extract features from textual items.  Article's textual features and metadata inputs were combined by using a sequence of Fully Connected (FC) layers to produce *Article Content Embeddings*.

For scalability reasons, *Article Content Embeddings* are not directly trained for recommendation task, but for a side task of news metadata classification. For this architecture instantiation of CHAMELEON, they were trained to classify the category (editorial section) of news articles.

After training, the *Article Content Embeddings* for news articles (NumPy matrix) are persisted into a Pickle dump file, for further usage by *NAR* module.

### Pre-processing data for the ACR module
Here is an example of the command for pre-processing articles text and metadata for the *ACR* module

It allows to specify the path of a CSV containing articles text and metadata (*input_articles_csv_path*), the path of pre-trained word embeddings (*input_word_embeddings_path*) in [Gensim format](https://radimrehurek.com/gensim/models/word2vec.html) and exports articles data into TFRecords format into *output_tf_records_path*, including the dictionaries that mapped tokenized words to sequences of int (*output_word_vocab_embeddings_path*) and metadata the categorical features encoders (*output_label_encoders*).

```bash
cd acr_module && \
DATA_DIR="[REPLACE BY THE GCOM ARTICLES DATASET PATH]" && \
python3 -m acr.preprocessing.acr_preprocess_gcom \
	--input_articles_csv_path ${DATA_DIR}/document_g1/documents_g1.csv \
 	--input_word_embeddings_path ${DATA_DIR}/word2vec/skip_s300.txt \
 	--vocab_most_freq_words 50000 \
 	--output_word_vocab_embeddings_path ${DATA_DIR}/pickles/acr_word_vocab_embeddings.pickle \
 	--output_label_encoders ${DATA_DIR}/pickles/acr_label_encoders.pickle \
 	--output_tf_records_path "${DATA_DIR}/articles_tfrecords/gcom_articles_tokenized_*.tfrecord.gz" \
 	--articles_by_tfrecord 5000
```
> From [acr_module/scripts/run_acr_preprocessing_gcom.sh](https://github.com/gabrielspmoreira/chameleon_recsys/blob/master/acr_module/scripts/run_acr_preprocessing_gcom.sh)

### Training ACR module
The ACR module can be trained either locally (example below) or using GCP ML Engine ([run_acr_training_gcom_mlengine.sh](https://github.com/gabrielspmoreira/chameleon_recsys/blob/master/acr_module/scripts/run_acr_training_gcom_mlengine.sh)).

The path of pre-processed TFRecords is informed in *train_set_path_regex* parameter, as well as the other *ACR* exported assets (*input_word_vocab_embeddings_path* and *input_label_encoders_path*). The trained *Article Content Embeddings* (NumPy matrix), with the dimensions specified by *acr_embeddings_size*, are exported (Pickle dump file) after training to *output_acr_metadata_embeddings_path*, for further usage by the *NAR* module.

```bash
cd acr_module && \
DATA_DIR="[REPLACE BY THE GCOM ARTICLES DATASET PATH]" && \
JOB_PREFIX=gcom && \
JOB_ID=`whoami`_${JOB_PREFIX}_`date '+%Y_%m_%d_%H%M%S'` && \
MODEL_DIR='/tmp/chameleon/gcom/jobs/'${JOB_ID} && \
echo 'Running training job and outputing to '${MODEL_DIR} && \
python3 -m acr.acr_trainer_gcom \
	--model_dir ${MODEL_DIR} \
	--train_set_path_regex "${DATA_DIR}/articles_tfrecords/gcom_articles_tokenized_*.tfrecord.gz" \
	--input_word_vocab_embeddings_path ${DATA_DIR}/pickles/acr_word_vocab_embeddings.pickle \
	--input_label_encoders_path ${DATA_DIR}/pickles/acr_label_encoders.pickle \
	--output_acr_metadata_embeddings_path ${DATA_DIR}/pickles/acr_articles_metadata_embeddings.pickle \
	--batch_size 64 \
	--truncate_tokens_length 300 \
	--training_epochs 5 \
	--learning_rate 3e-4 \
	--dropout_keep_prob 1.0 \
	--l2_reg_lambda 7e-4 \
	--text_feature_extractor "CNN" \
	--cnn_filter_sizes "3,4,5" \
	--cnn_num_filters 128 \
	--acr_embeddings_size 250
```
> From [run_acr_training_gcom_local.sh](https://github.com/gabrielspmoreira/chameleon_recsys/blob/master/acr_module/scripts/run_acr_training_gcom_local.sh)

## Next-Article Recommendation (NAR) module

The *Next-Article Recommendation (NAR)* module is responsible for providing news articles recommendations for active sessions.
Due to the high sparsity of users and their constant interests shift, the CHAMELEON instantiation reported in [1] leverages only session-based contextual information, ignoring possible users’ past sessions.

The inputs for the *NAR* module are: (1) the pre-trained *Article Content Embedding* of the last viewed article; (2) the contextual properties of the articles (popularity and recency); and (3) the user context (e.g. time, location, and device). These inputs are combined by Fully Connected layers to produce a *User-Personalized Contextual Article Embedding*, whose representations might differ for the same article, depending on the user context and on the current article context (popularity and recency).

The *NAR* module uses a type of Recurrent Neural Network (RNN) – the Long Short-Term Memory (LSTM) – to model the sequence of articles read by users in their sessions, represented by their *User-Personalized Contextual Article Embeddings*. For each article of the sequence, the RNN outputs a *Predicted Next-Article Embedding* – the expected representation of a news content the user would like to read next in the active session.

In most deep learning architectures proposed for RS, the neural network outputs a vector whose dimension is the number of
available items. Such approach may work for domains were the items number is more stable, like movies and books. Although, in
the dynamic scenario of news recommendations, where thousands of news stories are added and removed daily, such approach could require full retrain of the network, as often as new articles are published.  

For this reason, instead of using a softmax cross-entropy loss, the NAR module is trained to maximize the similarity between the *Predicted Next-Article Embedding* and the *User-Personalized Contextual Article Embedding* corresponding to the next article actually read by the user in his session (positive sample), whilst minimizing its similarity with negative samples (articles not read by the user during the session). With this strategy to deal with item cold-start, a newly published article might be immediately recommended, as soon as its *Article Content Embedding* is trained and added to a repository.

### Pre-processing data for the NAR module
The NAR module expects TFRecord files containing [SequenceExamples](https://www.tensorflow.org/api_docs/python/tf/train/SequenceExample) of user sessions's context and clicked articles.

The following example command ([run_nar_preprocessing_gcom_dlrs.sh](https://github.com/gabrielspmoreira/chameleon_recsys/blob/master/nar_module/scripts/run_nar_preprocessing_gcom_dlrs.sh)) takes the path of CSV files, containing users sessions split by hour (*input_clicks_csv_path_regex*), and outputs the corresponding TFRecord files.

```bash
cd nar_module && \
DATA_DIR="[REPLACE WITH THE PATH TO UNZIPPED GCOM DATASET FOLDER]" && \
python3 -m nar.preprocessing.nar_preprocess_gcom_dlrs \
--input_clicks_csv_path_regex "${DATA_DIR}/clicks/clicks_hour_*" \
--output_sessions_tfrecords_path "${DATA_DIR}/sessions_tfrecords/sessions_hour_*.tfrecord.gz"
```

### Training and evaluating the NAR module
The *NAR* module is trained and evaluated according to the following *Temporal Offline Evaluation Method*, described in [1]:
1. Train the NAR module with sessions within the active hour.
2. Evaluate the NAR module with sessions within the next hour, for the task of the next-click prediction.

The following baseline methods (described in more detail in [1]) are also trained and evaluated in parallel, as benchmarks for CHAMELEON accuracy:
- **Co-occurrent**
- **Sequential Rules (SR)**
- **Item-kNN**
- **Vector Multiplication Session-Based kNN (V-SkNN)**
- **Recently Popular**
- **Content-Based**

The choosen evaluation metrics were **Hit-Rate@N** and **MRR@N**.

#### Parameters
The *train_set_path_regex* parameter expects the path (local or GCS) where the sessions' TFRecords were exported. It also expects the path of the articles metadata CSV (*acr_module_articles_metadata_csv_path*) and the Pickle dump file with the *Article Content Embeddings* (*acr_module_articles_content_embeddings_pickle_path*).

It is necessary to specify a subset of files (representing sessions started in the same hour) for training and evaluation (*train_files_from* to *train_files_up_to*). The frequency of evaluation is specified in *training_hours_for_each_eval* (e.g. *training_hours_for_each_eval=5* means that after training on 5 hour's (files) sessions, the next hour (file) is used for evaluation. 

The *disable_eval_benchmarks* parameter disables training and evaluation of benchmark methods (useful for speed up). And the *save_eval_sessions_negative_samples* parameter saves a JSON lines file with the negative samples randomly sampled for each user session, for a consistent evaluation of the *GRU4Rec* benchmark (see next sections) which.


The *NAR* module can either be trained locally or using GCP ML Engine, as following examples.  

#### Local training and evaluation of NAR module


```bash
cd nar_module && \
DATA_DIR="[REPLACE WITH THE PATH TO UNZIPPED GCOM DATASET FOLDER]" && \
JOB_PREFIX=gcom && \
JOB_ID=`whoami`_${JOB_PREFIX}_`date '+%Y_%m_%d_%H%M%S'` && \
MODEL_DIR='/tmp/chameleon/jobs/'${JOB_ID} && \
echo 'Running training job and outputing to '${MODEL_DIR} && \
python3 -m nar.nar_trainer_gcom_dlrs \
	--model_dir ${MODEL_DIR} \
	--train_set_path_regex "${DATA_DIR}/sessions_tfrecords/sessions_hour_*.tfrecord.gz" \
	--train_files_from 0 \
	--train_files_up_to 72 \
	--training_hours_for_each_eval 5 \
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
	--eval_metrics_by_session_position \
	--save_eval_sessions_negative_samples
```
> From [run_nar_train_gcom_dlrs_local.sh](https://github.com/gabrielspmoreira/chameleon_recsys/blob/master/nar_module/scripts/run_nar_train_gcom_dlrs_local.sh)

#### ML Engine training and evaluation of NAR module
When training the *NAR* module in GCP ML Engine, two parameters are specially important:
- *use_local_cache_model_dir* - TensorFlow logs and summaries are usually large and saving them directly to GCS considerally slows down the training/evaluation process. When this parameter is present, TF model logs to a local folder in the worker machine.
- *save_results_each_n_evals* - The frequency (number of evaluation loops) in which TF logs saved locally are uploaded to a GCS path (*model_dir*).

P.s. As the proposed *Temporal Offline Evaluation Method* assumes that hours are trained and evaluated in sequence, a single ML Engine worker with GPU is used (*scale-tier=basic-gpu*) to avoid leaking from future sessions (due to the asynchronous training when more than one worker is used).

```bash
cd nar_module && \
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
```
> [run_nar_train_gcom_dlrs_mlengine.sh](https://github.com/gabrielspmoreira/chameleon_recsys/blob/master/nar_module/scripts/run_nar_train_gcom_dlrs_mlengine.sh)

#### GRU4Rec benchmark - Training and evaluation
The [GRU4Rec](https://github.com/hidasib/GRU4Rec) benchmark was originally developed by [2] in Theano. For this reason, its training and evaluation could not be run in parallel with CHAMELEON in GCP ML Engine.

The GRU4Rec evaluation process was adapted to use the same protocol, metrics and negative samples, randomly sampled during NAR training.

```bash
cd nar_module && \
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
```
> From [run_gru4rec_gcom_dlrs.sh](https://github.com/gabrielspmoreira/chameleon_recsys/blob/master/nar_module/scripts/run_gru4rec_gcom_dlrs.sh)

## References
[1] Gabriel de Souza Pereira Moreira, Felipe Ferreira, and Adilson Marques da Cunha. 2018. News Session-Based Recommendations using Deep Neural Networks. In 3rd Workshop on Deep Learning for Recommender Systems (DLRS 2018), October 6, 2018, Vancouver, BC, Canada. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3270323.3270328

[2] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based recommendations with recurrent neural networks. In Proceedings of Forth International Conference on Learning Representations, 2016.