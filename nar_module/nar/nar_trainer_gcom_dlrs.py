from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import time
import tensorflow as tf
import json
import os
import re
import numpy as np
import pandas as pd
import tempfile
import sys
import logging

from .utils import deserialize, resolve_files, chunks, merge_two_dicts, log_elapsed_time, append_lines_to_text_file
from .datasets import prepare_dataset_iterator
from .nar_model import ClickedItemsState, ItemsStateUpdaterHook, NARModuleModel
from .benchmarks import RecentlyPopularRecommender, ContentBasedRecommender, ItemCooccurrenceRecommender, ItemKNNRecommender, SessionBasedKNNRecommender, SequentialRulesRecommender

from .nar_utils import  load_nar_module_preprocessing_resources, save_eval_benchmark_metrics_csv, \
        upload_model_output_to_gcs, dowload_model_output_from_gcs


import glob        

tf.logging.set_verbosity(tf.logging.INFO)

RANDOM_SEED=42


#Model params
tf.flags.DEFINE_integer('batch_size', default=64, help='Batch size')
tf.flags.DEFINE_integer('truncate_session_length', default=20, help='Truncate long sessions to this max. size')
tf.flags.DEFINE_float('learning_rate', default=1e-3, help='Lerning Rate')
tf.flags.DEFINE_float('dropout_keep_prob', default=1.0, help='Dropout (keep prob.)')
tf.flags.DEFINE_float('reg_l2', default=0.0002, help='L2 regularization')
tf.flags.DEFINE_float('cosine_loss_gamma', default=2.0, help='Initial value for Gamma variable on cosine similarity')
tf.flags.DEFINE_integer('recent_clicks_buffer_size', default=500, help='Size of Recent Clicks Buffer')
tf.flags.DEFINE_integer('eval_metrics_top_n', default=3, help='Eval. metrics Top N')
tf.flags.DEFINE_integer('CAR_embedding_size', default=512, help='CAR submodule embedding size')
tf.flags.DEFINE_integer('rnn_units', default=1024, help='Number of units of RNN cell')
tf.flags.DEFINE_integer('rnn_num_layers', default=1, help='Number of of RNN layers')
tf.flags.DEFINE_integer('train_total_negative_samples', default=5, help='Total negative samples for training')
tf.flags.DEFINE_integer('train_negative_samples_from_buffer', default=10, help='Training Negative samples from recent clicks buffer')
tf.flags.DEFINE_integer('eval_total_negative_samples', default=20, help='Total negative samples for evaluation')
tf.flags.DEFINE_integer('eval_negative_samples_from_buffer', default=50, help='Eval. Negative samples from recent clicks buffer')
tf.flags.DEFINE_bool('save_histograms', default=False, help='Save histograms to view on Tensorboard (make job slower)')
tf.flags.DEFINE_bool('disable_eval_benchmarks', default=False, help='Disable eval benchmarks')
tf.flags.DEFINE_bool('eval_metrics_by_session_position', default=False, help='Computes eval metrics at each position within session (e.g. 1st click, 2nd click)')



#Control params
#tf.flags.DEFINE_string('data_dir', default_value='./tmp',
#                    help='Directory where the dataset is located')
tf.flags.DEFINE_string('train_set_path_regex',
                    default='/train*.tfrecord', help='Train set regex')
tf.flags.DEFINE_string('acr_module_articles_metadata_csv_path',
                    default='/pickles', help='ACR module''s articles metadata CSV')
tf.flags.DEFINE_string('acr_module_articles_content_embeddings_pickle_path',
                    default='/pickles', help='ACR module''s trained article content embeddings')
tf.flags.DEFINE_string('model_dir', default='./tmp',
                    help='Directory where save model checkpoints')
tf.flags.DEFINE_string('warmup_model_dir', default=None,
                    help='Directory where model checkpoints of a previous job where output, to warm start this network training')

tf.flags.DEFINE_integer('train_files_from', default=0, help='Train model starting from file N')
tf.flags.DEFINE_integer('train_files_up_to', default=100, help='Train model up to file N')
tf.flags.DEFINE_integer('training_hours_for_each_eval', default=5, help='Train model for N hours before evaluation of the next hour')
tf.flags.DEFINE_integer('save_results_each_n_evals', default=5, help='Saves to disk and uploads to GCS (ML Engine) the incremental evaluation results each N evaluations')
tf.flags.DEFINE_bool('save_eval_sessions_negative_samples', default=False, help='Save negative samples of each session during evaluation')
tf.flags.DEFINE_bool('use_local_cache_model_dir', default=False, help='Persists checkpoints and events in a local temp file, copying to GCS in the end of the process (useful for ML Engine jobs, because saving and loading checkpoints slows training job)')
#Default param used by ML Engine to validate whether the path exists
tf.flags.DEFINE_string('job-dir', default='./tmp', help='Job dir to save staging files')

FLAGS = tf.flags.FLAGS
#params_dict = tf.app.flags.FLAGS.flag_values_dict()
#tf.logging.info('PARAMS: {}'.format(json.dumps(params_dict)))

def get_articles_features_config():
    articles_features_config = {
        #Required fields
        'article_id': {'type': 'categorical', 'dtype': 'int', 'cardinality': 364047},
        'created_at_ts': {'type': 'numerical', 'dtype': 'int'},
        #Additional metadata fields
        'publisher_id': {'type': 'categorical', 'dtype': 'int', 'cardinality': 1},
        'category_id': {'type': 'categorical', 'dtype': 'int', 'cardinality': 461},
    }

    tf.logging.info('Article Features: {}'.format(articles_features_config))       
    return articles_features_config

def load_acr_module_resources(articles_metadata_csv_path, articles_content_embeddings_pickle_path):  
    
    content_article_embeddings = deserialize(articles_content_embeddings_pickle_path)    
    tf.logging.info("Read ACR article content embeddings: {}".format(content_article_embeddings.shape))

    articles_metadata_df = pd.read_csv(tf.gfile.Open(articles_metadata_csv_path))
    tf.logging.info("Read ACR articles metadata: {}".format(len(articles_metadata_df)))

    return articles_metadata_df, content_article_embeddings


def process_articles_metadata(articles_metadata_df, articles_features_config):
    articles_metadata = {}
    for feature_name in articles_features_config:
        articles_metadata[feature_name] = articles_metadata_df[feature_name].values
    return articles_metadata



def get_session_features_config():
    session_features_config = {
        'single_features': {
            #Control features
            'user_id': {'type': 'categorical', 'dtype': 'int', 'cardinality': 341193},
            'session_id': {'type': 'categorical', 'dtype': 'int'},
            'session_start': {'type': 'categorical', 'dtype': 'int'},
            'session_size': {'type': 'categorical', 'dtype': 'int'},
        },
        'sequence_features': {
            #Required sequence features
            'event_timestamp': {'type': 'numerical', 'dtype': 'int'},
            'item_clicked': {'type': 'categorical', 'dtype': 'int', 'cardinality': 364047},
            #Categorical features            
            'environment': {'type': 'categorical', 'dtype': 'int', 'cardinality': 5},
            'deviceGroup': {'type': 'categorical', 'dtype': 'int', 'cardinality': 6},
        }
    }
   
    tf.logging.info('Session Features: {}'.format(session_features_config))

    return session_features_config
    

def nar_module_model_fn(features, labels, mode, params):    
    #features_input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        negative_samples = params['train_total_negative_samples']
        negative_sample_from_buffer = params['train_negative_samples_from_buffer']
    elif mode == tf.estimator.ModeKeys.EVAL:
        negative_samples = params['eval_total_negative_samples']
        negative_sample_from_buffer = params['eval_negative_samples_from_buffer']

    
    dropout_keep_prob = params['dropout_keep_prob'] if mode == tf.estimator.ModeKeys.TRAIN else 1.0
    
    
    eval_metrics_top_n = params['eval_metrics_top_n']
    
    model = NARModuleModel(mode, features, labels,
              session_features_config=params['session_features_config'],
              articles_features_config=params['articles_features_config'],
              batch_size=params['batch_size'], 
              lr=params['lr'],
              keep_prob=dropout_keep_prob,
              negative_samples=negative_samples,
              negative_sample_from_buffer=negative_sample_from_buffer,
              reg_weight_decay=params['reg_weight_decay'], 
              cosine_loss_gamma=params['cosine_loss_gamma'], 
              articles_metadata=params['articles_metadata'],
              content_article_embeddings_matrix=params['content_article_embeddings_matrix'],
              recent_clicks_buffer_size=params['recent_clicks_buffer_size'],
              CAR_embedding_size=params['CAR_embedding_size'],
              rnn_units=params['rnn_units'],
              metrics_top_n=eval_metrics_top_n,
              plot_histograms=params['save_histograms']              
             )
    
    #Using these variables as global so that they persist across different train and eval
    global clicked_items_state, eval_sessions_metrics_log, sessions_negative_items_log

    eval_benchmark_classifiers = []
    if not FLAGS.disable_eval_benchmarks:
        eval_benchmark_classifiers=[{'recommender': RecentlyPopularRecommender, 'params': {}},
                                    {'recommender': ItemCooccurrenceRecommender, 'params': {}},
                                    {'recommender': ItemKNNRecommender, 'params': {}},
                                    {'recommender': SessionBasedKNNRecommender, 
                                          'params': {'sessions_buffer_size': 3000, #Buffer size of last processed sessions
                                                     'candidate_sessions_sample_size': 200, #Number of candidate near sessions to sample
                                                     'sampling_strategy': 'recent', #(recent,random)
                                                     'nearest_neighbor_session_for_scoring': 50, #Nearest neighbors to compute item scores      
                                                     'similarity': 'jaccard', #(jaccard, cosine)
                                                     'first_session_clicks_decay': 'div' #Decays weight of first user clicks in active session when finding neighbor sessions (same, div, linear, log, quadradic)
                                                     }},
                                    {'recommender': ContentBasedRecommender, 
                                          'params': {'articles_metadata': params['articles_metadata'],
                                                     'content_article_embeddings_matrix': params['content_article_embeddings_matrix']}},
                                    {'recommender': SequentialRulesRecommender,
                                          'params': {'max_clicks_dist': 10, #Max number of clicks to walk back in the session from the currently viewed item. (Default value: 10) 
                                                     'dist_between_clicks_decay': 'div' #Decay function for distance between two items clicks within a session (linear, same, div, log, qudratic). (Default value: div) 
                                                     }}
                                   ]
                                                              
    hooks = [ItemsStateUpdaterHook(mode, model, 
                                   eval_metrics_top_n=eval_metrics_top_n,
                                   clicked_items_state=clicked_items_state, 
                                   eval_sessions_metrics_log=eval_sessions_metrics_log,
                                   sessions_negative_items_log=sessions_negative_items_log,
                                   eval_benchmark_classifiers=eval_benchmark_classifiers,
                                   eval_metrics_by_session_position=params['eval_metrics_by_session_position']
                                   )] 
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        return tf.estimator.EstimatorSpec(mode, loss=model.total_loss, train_op=model.train,
                                      training_chief_hooks=hooks)
    elif mode == tf.estimator.ModeKeys.EVAL:  

        eval_metrics = {'hitrate_at_1': (model.next_item_accuracy_at_1, model.next_item_accuracy_at_1_update_op),
                        'hitrate_at_n': (model.recall_at_n, model.recall_at_n_update_op),
                        'mrr_at_n': (model.mrr, model.mrr_update_op),   
                        #'ndcg_at_n': (model.ndcg_at_n_mean, model.ndcg_at_n_mean_update_op),                 
                       }
                        
        return tf.estimator.EstimatorSpec(mode, loss=model.total_loss, eval_metric_ops=eval_metrics,
                                      evaluation_hooks=hooks) 


def build_estimator(model_dir,
    content_article_embeddings_matrix, 
    articles_metadata, articles_features_config,
    session_features_config):
    """Build an estimator appropriate for the given model type."""

    run_config = tf.estimator.RunConfig(tf_random_seed=RANDOM_SEED,
                                        keep_checkpoint_max=1, 
                                        save_checkpoints_secs=1200, 
                                        save_summary_steps=100)

    estimator = tf.estimator.Estimator(
        config=run_config,
        model_dir=model_dir,
        model_fn=nar_module_model_fn,    
        params={
            'batch_size': FLAGS.batch_size,
            'lr': FLAGS.learning_rate,
            'dropout_keep_prob': FLAGS.dropout_keep_prob,
            'reg_weight_decay': FLAGS.reg_l2,
            'recent_clicks_buffer_size': FLAGS.recent_clicks_buffer_size,
            'eval_metrics_top_n': FLAGS.eval_metrics_top_n,
            'CAR_embedding_size': FLAGS.CAR_embedding_size,
            'rnn_units': FLAGS.rnn_units,
            'train_total_negative_samples': FLAGS.train_total_negative_samples,
            'train_negative_samples_from_buffer': FLAGS.train_negative_samples_from_buffer,
            'eval_total_negative_samples': FLAGS.eval_total_negative_samples,
            'eval_negative_samples_from_buffer': FLAGS.eval_negative_samples_from_buffer,
            'cosine_loss_gamma': FLAGS.cosine_loss_gamma,
            'save_histograms': FLAGS.save_histograms,
            'eval_metrics_by_session_position': FLAGS.eval_metrics_by_session_position,

            #From pre-processing
            'session_features_config': session_features_config,
            'articles_features_config': articles_features_config,
            'articles_metadata': articles_metadata,            
            #From ACR module
            'content_article_embeddings_matrix': content_article_embeddings_matrix
        })

    return estimator


#Saving the negative samples used to evaluate each sessions, so that benchmarks metrics outside the framework (eg. Matrix Factorization) can be comparable
def save_sessions_negative_items(model_output_dir, sessions_negative_items_list):
    append_lines_to_text_file(os.path.join(model_output_dir, 'eval_sessions_negative_samples.json'), 
                                           map(lambda x: json.dumps({'session_id': x['session_id'],
                                                                     'negative_items': x['negative_items'].tolist()}), 
                                               sessions_negative_items_list))


#Global vars updated by the Estimator Hook
clicked_items_state = None
eval_sessions_metrics_log = [] 
sessions_negative_items_log = [] if FLAGS.save_eval_sessions_negative_samples else None

def main(unused_argv):
    try:
        # Capture whether it will be a single training job or a hyper parameter tuning job.
        tf_config_env = json.loads(os.environ.get('TF_CONFIG', '{}'))
        task_data = tf_config_env.get('task') or {'type': 'master', 'index': 0}
        trial = task_data.get('trial')

        running_on_mlengine = (len(tf_config_env) > 0)
        print('Running {}'.format('on Google ML Engine' if running_on_mlengine else 'on a server/machine'))

        #Disabling duplicate logs on console when running locally
        logging.getLogger('tensorflow').propagate = running_on_mlengine

        tf.logging.info('Starting training job')    

        gcs_model_output_dir = FLAGS.model_dir
        #If must persist and load model ouput in a local cache (to speedup in ML Engine)
        if FLAGS.use_local_cache_model_dir:
            model_output_dir = tempfile.mkdtemp()
            tf.logging.info('Created local temp folder for models output: {}'.format(model_output_dir))
        else:
            model_output_dir = gcs_model_output_dir

        if trial is not None:
            model_output_dir = os.path.join(model_output_dir, trial)
            gcs_model_output_dir = os.path.join(gcs_model_output_dir, trial)
            tf.logging.info(
                "Hyperparameter Tuning - Trial {} - model_dir = {} - gcs_model_output_dir = {} ".format(trial, model_output_dir, gcs_model_output_dir))

        tf.logging.info('Will save temporary model outputs to {}'.format(model_output_dir))

        #If should warm start training from other previously trained model
        if FLAGS.warmup_model_dir != None:
            tf.logging.info('Copying model outputs from previous job ({}) for warm start'.format(FLAGS.warmup_model_dir))
            dowload_model_output_from_gcs(model_output_dir, 
                                          gcs_model_dir=FLAGS.warmup_model_dir,
                                          files_pattern=['graph.pb', 
                                                         'model.ckpt-', 
                                                         'checkpoint'])

            local_files_after_download_to_debug = list(glob.iglob("{}/**/*".format(model_output_dir), recursive=True))
            tf.logging.info('Files copied from GCS to warm start training: {}'.format(local_files_after_download_to_debug))

        tf.logging.info('Loading ACR module assets')
        articles_metadata_df, content_article_embeddings_matrix = \
                load_acr_module_resources(FLAGS.acr_module_articles_metadata_csv_path, 
                                          FLAGS.acr_module_articles_content_embeddings_pickle_path)

        articles_features_config = get_articles_features_config()
        articles_metadata = process_articles_metadata(articles_metadata_df, articles_features_config)
        
        session_features_config = get_session_features_config()
 
        tf.logging.info('Building NAR model')
        global eval_sessions_metrics_log, clicked_items_state, sessions_negative_items_log
        eval_sessions_metrics_log = []
        clicked_items_state = ClickedItemsState(FLAGS.recent_clicks_buffer_size, content_article_embeddings_matrix.shape[0])
        model = build_estimator(model_output_dir, 
            content_article_embeddings_matrix, articles_metadata, articles_features_config,
            session_features_config)
        
        tf.logging.info('Getting training file names')
        train_files = resolve_files(FLAGS.train_set_path_regex)

        if FLAGS.train_files_from > FLAGS.train_files_up_to:
            raise Exception('Final training file cannot be lower than Starting training file')
        train_files = train_files[FLAGS.train_files_from:FLAGS.train_files_up_to+1]

        tf.logging.info('{} files where the network will be trained and evaluated on, from {} to {}' \
                            .format(len(train_files), train_files[0], train_files[-1]))

        start_train = time()
        tf.logging.info("Starting Training Loop")

        #training_chunks_count = int(len(train_files) / float(FLAGS.training_hours_for_each_eval))
        training_files_chunks = list(chunks(train_files, FLAGS.training_hours_for_each_eval))

        for chunk_id in range(0, len(training_files_chunks)):     

            training_files_chunk = training_files_chunks[chunk_id]
            tf.logging.info('Training files from {} to {}'.format(training_files_chunk[0], training_files_chunk[-1]))
            model.train(input_fn=lambda: prepare_dataset_iterator(training_files_chunk, session_features_config, 
                                                                          batch_size=FLAGS.batch_size,
                                                                          truncate_session_length=FLAGS.truncate_session_length))
            
            if chunk_id < len(training_files_chunks)-1:
                #Using the first hour of next training chunck as eval
                eval_file = training_files_chunks[chunk_id+1][0]
                tf.logging.info('Evaluating file {}'.format(eval_file))
                model.evaluate(input_fn=lambda: prepare_dataset_iterator(eval_file, session_features_config, 
                                                                                 batch_size=FLAGS.batch_size,
                                                                                 truncate_session_length=FLAGS.truncate_session_length))

            #After each number of train/eval loops
            if chunk_id % FLAGS.save_results_each_n_evals == 0:
                tf.logging.info('Saving eval metrics')
                save_eval_benchmark_metrics_csv(eval_sessions_metrics_log, model_output_dir,
                                        training_hours_for_each_eval=FLAGS.training_hours_for_each_eval)

                if FLAGS.save_eval_sessions_negative_samples:
                    #Flushing to disk the negative samples used to evaluate each sessions, 
                    #so that benchmarks metrics outside the framework (eg. Matrix Factorization) can be comparable
                    save_sessions_negative_items(model_output_dir, sessions_negative_items_log)
                    sessions_negative_items_log = []

                #If must persist and load model ouput in a local cache (to speedup in ML Engine)
                if FLAGS.use_local_cache_model_dir:
                    tf.logging.info('Uploading cached results to GCS')
                    upload_model_output_to_gcs(model_output_dir, gcs_model_dir=gcs_model_output_dir,
                                               files_pattern=['events.out.tfevents.','.csv', '.json'])



        tf.logging.info('Finalized Training')

        save_eval_benchmark_metrics_csv(eval_sessions_metrics_log, model_output_dir,
                                        training_hours_for_each_eval=FLAGS.training_hours_for_each_eval)

        if FLAGS.save_eval_sessions_negative_samples:
            #Flushing to disk the negative samples used to evaluate each sessions, 
            #so that benchmarks metrics outside the framework (eg. Matrix Factorization) can be comparable
            save_sessions_negative_items(model_output_dir, sessions_negative_items_log)
            sessions_negative_items_log = []

        tf.logging.info('Saved eval metrics')

        #If must persist and load model ouput in a local cache (to speedup in ML Engine)
        if FLAGS.use_local_cache_model_dir:
            upload_model_output_to_gcs(model_output_dir, gcs_model_dir=gcs_model_output_dir,
                                        files_pattern=None)
            

        log_elapsed_time(start_train, 'Finalized TRAINING Loop')
    
    except Exception as ex:
        tf.logging.error('ERROR: {}'.format(ex))
        raise



if __name__ == '__main__':  
    tf.app.run()    