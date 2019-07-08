import logging
import os
from time import time
import json
import tempfile
import tensorflow as tf
import numpy as np
import pandas as pd

from .utils import resolve_files, deserialize, serialize, log_elapsed_time
from .acr_model import ACR_Model
from .acr_datasets import prepare_dataset


tf.logging.set_verbosity(tf.logging.INFO)

#Making results reproduceable
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)


#Control params
#tf.flags.DEFINE_string('data_dir', default='',
#                    help='Directory where the dataset is located')
tf.flags.DEFINE_string('train_set_path_regex',
                    default='/train*.tfrecord', help='Train set regex')
tf.flags.DEFINE_string('model_dir', default='./tmp',
                    help='Directory where save model checkpoints')

tf.flags.DEFINE_string('input_word_vocab_embeddings_path', default='',
                    help='Input path for a pickle with words vocabulary and corresponding word embeddings')
tf.flags.DEFINE_string('input_label_encoders_path', default='',
                    help='Input path for a pickle with label encoders (article_id, category_id, publisher_id)')
tf.flags.DEFINE_string('output_acr_metadata_embeddings_path', default='',
                    help='Output path for a pickle with articles metadata and content embeddings')

#Model params
tf.flags.DEFINE_string('text_feature_extractor', default="CNN", help='Feature extractor of articles text: (CNN | LSTM | GRU')
tf.flags.DEFINE_string('training_task', default="metadata_classification", help='Training task: (metadata_classification | autoencoder)')

tf.flags.DEFINE_float('autoencoder_noise', default=0.0, help='Adds white noise with this standard deviation to the input word embeddings')


tf.flags.DEFINE_string('cnn_filter_sizes', default="3,4,5", help='CNN layers filter sizes (sliding window over words)')
tf.flags.DEFINE_integer('cnn_num_filters', default=128, help='Number of filters of CNN layers')
tf.flags.DEFINE_integer('rnn_units', default=250, help='Number of units in each RNN layer')
tf.flags.DEFINE_integer('rnn_layers', default=1, help='Number of RNN layers')

tf.flags.DEFINE_string('rnn_direction', default='unidirectional', help='Direction of RNN layers: (unidirectional | bidirectional)')


tf.flags.DEFINE_integer('acr_embeddings_size', default=250, help='Embedding size of output ACR embeddings')


#Training params
tf.flags.DEFINE_integer('batch_size', default=64, help='Batch size')
tf.flags.DEFINE_integer('truncate_tokens_length', default=300, help='Truncate the sequence of tokens (words) to this limit')
tf.flags.DEFINE_integer('training_epochs', default=10, help='Training epochs')
tf.flags.DEFINE_float('learning_rate', default=1e-3, help='Lerning Rate')
tf.flags.DEFINE_float('dropout_keep_prob', default=1.0, help='Dropout (keep prob.)')
tf.flags.DEFINE_float('l2_reg_lambda', default=1e-3, help='L2 regularization')


FLAGS = tf.flags.FLAGS
#params_dict = tf.app.flags.FLAGS.flag_values_dict()
#tf.logging.info('PARAMS: {}'.format(json.dumps(params_dict))) 


def get_session_features_config(acr_label_encoders):
    features_config = {
                'single_features': 
                    {'article_id': {'type': 'categorical', 'dtype': 'int'},
                    'publisher_id': {'type': 'categorical', 'dtype': 'int'},
                    'category_id': {'type': 'categorical', 'dtype': 'int'},
                    'created_at_ts': {'type': 'numerical', 'dtype': 'int'},
                    'text_length': {'type': 'numerical', 'dtype': 'int'},
                    },
                'sequence_features': {
                    'text': {'type': 'numerical', 'dtype': 'int'},
                    },
                'label_features': {
                    'category_id': {'type': 'categorical', 'dtype': 'int', 'classification_type': 'multiclass'}
                    }
                }

    #Adding cardinality to categorical features
    for feature_groups_key in features_config:
        features_group_config = features_config[feature_groups_key]
        for feature_name in features_group_config:
            if feature_name in acr_label_encoders and features_group_config[feature_name]['type'] == 'categorical':
                features_group_config[feature_name]['cardinality'] = len(acr_label_encoders[feature_name].classes_)

    tf.logging.info('Session Features: {}'.format(features_config))

    return features_config


def load_acr_preprocessing_assets(acr_label_encoders_path, word_vocab_embeddings_path):
    acr_label_encoders = deserialize(acr_label_encoders_path)
    tf.logging.info("Read article id label encoder: {}".format(len(acr_label_encoders['article_id'].classes_)))  

    (word_vocab, word_embeddings_matrix) = deserialize(word_vocab_embeddings_path)
    tf.logging.info("Read word embeddings: {}".format(word_embeddings_matrix.shape))  

    return acr_label_encoders, word_embeddings_matrix


def acr_model_fn(features, labels, mode, params):  
    publisher_column = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(
            key='publisher_id',
            num_buckets=params['features_config']['single_features']['publisher_id']['cardinality']))
    metadata_feature_columns = [publisher_column]
    metadata_features={'publisher_id': features['publisher_id']}

    acr_model = ACR_Model(params['training_task'], params['text_feature_extractor'], features, metadata_features, metadata_feature_columns, 
                         labels, params['features_config']['label_features'],
                         mode, params)   
    
    loss = None
    if (mode == tf.estimator.ModeKeys.TRAIN or
        mode == tf.estimator.ModeKeys.EVAL):
        loss = acr_model.total_loss
        
    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = acr_model.train_op

    eval_metrics = {}
    if (mode == tf.estimator.ModeKeys.TRAIN or
        mode == tf.estimator.ModeKeys.EVAL):
        eval_metrics = acr_model.eval_metrics
       
    predictions = None
    prediction_hooks = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {#Trained ACR embeddings
                       'acr_embedding': acr_model.article_content_embedding,
                       
                       #Additional metadata
                       'article_id': features['article_id'],                       
                       'category_id': features['category_id'],
                       'publisher_id': features['publisher_id'],
                       'created_at_ts': features['created_at_ts'],
                       'text_length': features['text_length']
                       }

        if params['training_task'] == 'autoencoder':
            predictions['input_text'] = features['text']
            predictions['predicted_word_ids'] = acr_model.predicted_word_ids
        elif params['training_task'] == 'metadata_classification':
            #Saves predicted categories
            for feature_name in acr_model.labels_predictions:
                predictions["predictions-{}".format(feature_name)] = acr_model.labels_predictions[feature_name]
     
    training_hooks = []   
    if params['enable_profiler_hook']:
        profile_hook = tf.train.ProfilerHook(save_steps=100,
                                    save_secs=None,
                                    show_dataflow=True,
                                    show_memory=False)
        training_hooks=[profile_hook]
    

    
    return tf.estimator.EstimatorSpec(
              mode=mode,
              predictions=predictions,
              loss=loss,
              train_op=train_op,
              eval_metric_ops=eval_metrics,
              training_hooks=training_hooks
              #prediction_hooks=prediction_hooks,
              )

def build_acr_estimator(model_output_dir, word_embeddings_matrix, features_config, special_token_embedding_vector):

    params = {'training_task': FLAGS.training_task,
              'text_feature_extractor': FLAGS.text_feature_extractor,  
              'word_embeddings_matrix': word_embeddings_matrix,              
              'vocab_size': word_embeddings_matrix.shape[0],
              'word_embedding_size': word_embeddings_matrix.shape[1],
              'cnn_filter_sizes': FLAGS.cnn_filter_sizes,
              'rnn_units': FLAGS.rnn_units,
              'rnn_layers': FLAGS.rnn_layers,
              'rnn_direction': FLAGS.rnn_direction,
              'cnn_num_filters': FLAGS.cnn_num_filters,
              'dropout_keep_prob': FLAGS.dropout_keep_prob,
              'l2_reg_lambda': FLAGS.l2_reg_lambda,
              'learning_rate': FLAGS.learning_rate,
              'acr_embeddings_size': FLAGS.acr_embeddings_size,
              'features_config': features_config,
              'special_token_embedding_vector': special_token_embedding_vector,
              'autoencoder_noise': FLAGS.autoencoder_noise,
              'enable_profiler_hook': False
              }

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,
                                    #device_count={'GPU': 0}
                                    )

    run_config = tf.estimator.RunConfig(tf_random_seed=RANDOM_SEED,
                                        save_summary_steps=100,
                                        keep_checkpoint_max=1,
                                        session_config=session_config
                                       )

    acr_cnn_classifier = tf.estimator.Estimator(model_fn=acr_model_fn,
                                            model_dir=model_output_dir,
                                            params=params, 
                                            config=run_config)

    return acr_cnn_classifier


def export_acr_metadata_embeddings(acr_label_encoders, articles_metadata_df, content_article_embeddings, output_path):    
    tf.logging.info('Exporting ACR Label Encoders, Article metadata and embeddings to {}'.format(output_path))
    to_serialize = (acr_label_encoders, articles_metadata_df, content_article_embeddings)
    serialize(output_path, to_serialize)


def get_articles_metadata_embeddings(article_metadata_with_pred_embeddings, ace_column='acr_embedding'):
    articles_metadata_df = pd.DataFrame(article_metadata_with_pred_embeddings).sort_values(by='article_id')


    #Checking whether article ids are sorted and contiguous
    assert (articles_metadata_df['article_id'].head(1).values[0] == 0)
    assert (len(articles_metadata_df) == articles_metadata_df['article_id'].tail(1).values[0]+1)

    content_article_embeddings = np.vstack(articles_metadata_df[ace_column].values)

    
    cols_to_export = ['article_id', 'category_id', 'created_at_ts', 'publisher_id', 'text_length']
    if FLAGS.training_task == 'autoencoder': 
        cols_to_export.extend(['input_text', 'predicted_word_ids'])
    elif FLAGS.training_task == 'metadata_classification': 
        cols_to_export.extend([col for col in articles_metadata_df.columns if col.startswith('predictions-')])


    #Filtering metadata columns to export
    articles_metadata_df = articles_metadata_df[cols_to_export]
                                                        

    #return articles_metadata_df, (content_article_embeddings, predicted_word_ids)
    return articles_metadata_df, content_article_embeddings

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


        start_train = time()
        tf.logging.info('Starting training job')

        model_output_dir = FLAGS.model_dir

        if trial is not None:
            model_output_dir = os.path.join(model_output_dir, trial)
            tf.logging.info(
                "Hyperparameter Tuning - Trial {}. model_dir = {}".format(trial, model_output_dir))
        else:
            tf.logging.info('Saving model outputs to {}'.format(model_output_dir))

        tf.logging.info('Loading ACR preprocessing assets')
        acr_label_encoders, word_embeddings_matrix = \
            load_acr_preprocessing_assets(FLAGS.input_label_encoders_path, 
                                          FLAGS.input_word_vocab_embeddings_path) 

        features_config = get_session_features_config(acr_label_encoders)          

        #input_tfrecords = os.path.join(FLAGS.data_dir, FLAGS.train_set_path_regex)
        input_tfrecords = FLAGS.train_set_path_regex
        tf.logging.info('Defining input data (TFRecords): {}'.format(input_tfrecords))


        #Creating an ambedding for a special token to initiate decoding of RNN-autoencoder
        special_token_embedding_vector = np.random.uniform(low=-0.04, high=0.04, 
                                                 size=[1,word_embeddings_matrix.shape[1]])

        acr_model = build_acr_estimator(model_output_dir, 
                                        word_embeddings_matrix, 
                                        features_config,
                                        special_token_embedding_vector)


        tf.logging.info('Training model')
        train_files = resolve_files(input_tfrecords)
        #files_to_train = train_files
        #To test generalization
        files_to_train = train_files#[:-10]
        print("Training articles on TFRecords from {} to {}".format(files_to_train[0],
                                                                    files_to_train[-1]))
        acr_model.train(input_fn=lambda: prepare_dataset(files=files_to_train,
                                                  features_config=features_config,
                                                  batch_size=FLAGS.batch_size, 
                                                  epochs=FLAGS.training_epochs, 
                                                  shuffle_dataset=True, 
                                                  shuffle_buffer_size=5000,
                                                  truncate_tokens_length=FLAGS.truncate_tokens_length))

        
        if FLAGS.training_task == 'metadata_classification':
            #The objective is to overfitting this network, so that the ACR embedding represent well the articles content
            tf.logging.info('Evaluating model')
            #Taking last train files as eval
            files_to_eval = train_files[-10:]
            eval_results = acr_model.evaluate(input_fn=lambda: prepare_dataset(files=files_to_eval,
                                                        features_config=features_config,
                                                        batch_size=FLAGS.batch_size, 
                                                        epochs=1, 
                                                        shuffle_dataset=False,
                                                        truncate_tokens_length=FLAGS.truncate_tokens_length))
            tf.logging.info('Evaluation results with TRAIN SET (objective is to overfit): {}'.format(eval_results))
        


        tf.logging.info('Predicting ACR embeddings')
        
        article_metadata_with_pred_embeddings = list(acr_model.predict(input_fn=lambda: prepare_dataset(files=train_files,
                                                    features_config=features_config,
                                                    batch_size=16, #Limits batch size to avoid OOM in the op to the predicted words for each word in the RNN auto-encoder (because of the vocabulary size is over 43k)    #FLAGS.batch_size, 
                                                    epochs=1, 
                                                    shuffle_dataset=False,
                                                    truncate_tokens_length=FLAGS.truncate_tokens_length)))

        

        articles_metadata_df, content_article_embeddings = get_articles_metadata_embeddings(article_metadata_with_pred_embeddings, 
                                                                                            ace_column='acr_embedding')
        tf.logging.info('Generated ACR embeddings (shape): {}'.format(content_article_embeddings.shape))   
        tf.logging.info('Generated ACR metadata (qtde): {}'.format(len(articles_metadata_df)))   
 
        output_path = FLAGS.output_acr_metadata_embeddings_path
        export_acr_metadata_embeddings(acr_label_encoders, articles_metadata_df, content_article_embeddings, output_path)


        log_elapsed_time(start_train, 'Finalized TRAINING')
    
    except Exception as ex:
        tf.logging.error('ERROR: {}'.format(ex))
        raise

if __name__ == '__main__':  
    tf.app.run()       