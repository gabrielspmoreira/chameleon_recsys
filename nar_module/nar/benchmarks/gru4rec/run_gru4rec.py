import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile

from .gru4rec2 import GRU4Rec
from .gru4rec2_evaluation import evaluate_sessions_batch_neg_samples

from ...datasets import make_dataset
from ...utils import resolve_files, chunks
from ...metrics import HitRate, MRR
from ...nar_utils import save_eval_benchmark_metrics_csv

import theano.misc.pkl_utils as pkl

MODEL_FILE = "gru_model.mdl"
EVAL_METRICS_FILE = 'eval_stats_gru4rec.csv'

#Configuring Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--train_set_path_regex", type=str, help="", default='')
parser.add_argument("--eval_sessions_negative_samples_json_path", type=str, help="", default='')
parser.add_argument("--training_hours_for_each_eval", type=int, help="", default=1)
parser.add_argument("--warmup_model_dump_file", type=str, help="", default=None)
parser.add_argument("--eval_metrics_top_n", type=int, help="", default=5)
parser.add_argument("--batch_size", type=int, help="", default=128)
parser.add_argument("--n_epochs", type=int, help="", default=3)
parser.add_argument("--optimizer", type=str, help="", default='adagrad')
parser.add_argument("--dropout_p_hidden", type=float, help="", default=0.0)
parser.add_argument("--learning_rate", type=float, help="", default=1e-4)
parser.add_argument("--l2_lambda", type=float, help="", default=1e-3)
parser.add_argument("--momentum", type=float, help="", default=0.0)
parser.add_argument("--embedding", type=int, help="", default=0)
ARGS = parser.parse_args()

#Disabling TF logs in console
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.logging.set_verbosity(tf.logging.ERROR)


def get_session_features_config():
    session_features_config = {
        'single_features': {
            ##Control features
            'user_id': {'type': 'categorical', 'dtype': 'int'},
            #'user_id': {'type': 'categorical', 'dtype': 'string'},
            'session_id': {'type': 'categorical', 'dtype': 'int'},
            #'session_id': {'type': 'categorical', 'dtype': 'string'},            
            'session_start': {'type': 'categorical', 'dtype': 'int'},
            'session_size': {'type': 'categorical', 'dtype': 'int'},
        },
        'sequence_features': {
            #Required sequence features
            'event_timestamp': {'type': 'categorical', 'dtype': 'int'},
            'item_clicked': {'type': 'categorical', 'dtype': 'int'},            
        }
    }

    return session_features_config


def prepare_dataset_iterator_local(features_config, batch_size=128, 
                                        truncate_session_length=20):
    with tf.device('/cpu:0'):
        files_placeholder = tf.placeholder(tf.string)

        # Make a dataset 
        ds = make_dataset(files_placeholder, features_config, batch_size=batch_size,
                            truncate_sequence_length=truncate_session_length)

        
        # Define an abstract iterator that has the shape and type of our datasets
        iterator = tf.data.Iterator.from_structure(ds.output_types,
                                                   ds.output_shapes)

        # This is an op that gets the next element from the iterator
        next_element = iterator.get_next()
        
        # These ops let us switch and reinitialize every time we finish an epoch    
        iterator_init_op = iterator.make_initializer(ds)

        return next_element, iterator_init_op, files_placeholder

def load_eval_negative_samples():
    eval_sessions_neg_samples_df = pd.read_json(ARGS.eval_sessions_negative_samples_json_path, lines=True,
                                                 dtype={'session_id': np.int64})
    eval_sessions_neg_samples = dict(eval_sessions_neg_samples_df[['session_id', 'negative_items']].values)
    return eval_sessions_neg_samples


def load_gru4rec_dataframe(next_element_op, iterator_init_op, files_placeholder,
                            training_files_chunk):
    data = []

    session_cnt = 0
    repeated = 0
    with tf.Session() as sess:  
        sess.run(iterator_init_op, feed_dict={files_placeholder: training_files_chunk})
        while True:  
            try:
                #One session by batch
                batch_inputs, batch_labels = sess.run(next_element_op)

                item_ids_session = set()

                session_id = batch_inputs['session_id'][0]
                for item, ts in zip(batch_inputs['item_clicked'][0], batch_inputs['event_timestamp'][0]):
                    if item in item_ids_session:
                        repeated += 1
                    item_ids_session.add(item)
                    data.append((session_id, item, ts))

                #Adding last item (label)
                last_item = batch_labels['label_last_item'][0][0]
                if last_item in item_ids_session:
                    repeated += 1
                data.append((session_id, last_item, ts))

                session_cnt += 1
                #if cnt % 100 == 0:
                #    print("Sessions processed: {} - Clicks: {}".format(session_cnt, len(data)))
            except tf.errors.OutOfRangeError as e:
                break

    if len(data) > 0:
        print("Sessions read: {} - Clicks: {}".format(session_cnt, len(data)))
    else:
        print('WARNING: NO DATA FOUND!')
    data_df = pd.DataFrame(data, columns=['SessionId', 'ItemId', 'Time'])
    return data_df

if __name__ == '__main__':

    temp_folder = tempfile.mkdtemp()
    print('Creating temporary folder: {}'.format(temp_folder))


    streaming_metrics = [metric(topn=ARGS.eval_metrics_top_n) for metric in [HitRate, MRR]]

    if ARGS.warmup_model_dump_file:
        print('Loading pre-trained GRU model from: {}'.format(ARGS.warmup_model_dump_file))

        gru_file = open(ARGS.warmup_model_dump_file, "rb+")
        try:
            gru = pkl.load(gru_file)
        finally:
            gru_file.close()

    else:      
        gru = GRU4Rec(n_epochs=ARGS.n_epochs, loss='bpr-max-0.5', final_act='linear', 
            hidden_act='tanh', layers=[300], adapt=ARGS.optimizer, decay=0.0, 
            batch_size=ARGS.batch_size, 
            dropout_p_embed=0.0, dropout_p_hidden=ARGS.dropout_p_hidden, 
            learning_rate=ARGS.learning_rate, 
            lmbd=ARGS.l2_lambda, momentum=ARGS.momentum, n_sample=200, 
            sample_alpha=0.0, time_sort=False, embedding=ARGS.embedding, #grad_cap=2.0,
            session_key='SessionId', item_key='ItemId', time_key='Time')

    try:
        hit_rates = []
        mrrs = []
            
        eval_sessions_neg_samples = load_eval_negative_samples()

        train_files = resolve_files(ARGS.train_set_path_regex)
        training_files_chunks = list(chunks(train_files, ARGS.training_hours_for_each_eval))
        session_features_config = get_session_features_config()

        

        eval_sessions_metrics_log = []
        

        next_element_op, iterator_init_op, files_placeholder = prepare_dataset_iterator_local(
                                    session_features_config, 
                                 batch_size=1)

        print("Starting Training Loop")
        for chunk_id in range(0, len(training_files_chunks)-1):
            
            training_files_chunk = training_files_chunks[chunk_id]
            print('Training from file {} to {}'.format(training_files_chunk[0], training_files_chunk[-1]))
            train_df = load_gru4rec_dataframe(next_element_op, iterator_init_op, files_placeholder,
                                                training_files_chunk)

            if train_df['SessionId'].nunique() < ARGS.batch_size:
                print('WARNING: Ignoring training file for having less than {} sessions: {}'.format(ARGS.batch_size, train_df['SessionId'].nunique()))
                print('')
                continue

            gru.fit(train_df, retrain=hasattr(gru, 'n_items'), sample_store=0)
            


            #Using the first hour of next training chunck as eval
            eval_file = training_files_chunks[chunk_id+1][0]

            print('Evaluating file {}'.format(eval_file))
            test_df = load_gru4rec_dataframe(next_element_op, iterator_init_op, files_placeholder,
                                                eval_file)

            #IMPORTANT: Filtering for prediction only items present in train set, to avoid errors on GRU4REC
            test_df = pd.merge(test_df, pd.DataFrame({'ItemIdx': gru.itemidmap.values, 
                                                      'ItemId': gru.itemidmap.index}), on='ItemId', how='inner')
            test_df = test_df.groupby('SessionId').filter(lambda x: len(x) > 1)

            if test_df['SessionId'].nunique() < ARGS.batch_size:
                print('WARNING: Ignoring test file for having less than {} sessions: {}'.format(ARGS.batch_size, test_df['SessionId'].nunique()))
                eval_sessions_metrics_log.append({'hitrate_at_n_gru4rec': None,
                                              'mrr_at_n_gru4rec': None,
                                              'clicks_count': len(test_df),
                                              'sessions_count': test_df['SessionId'].nunique()})
                print('')
                continue

            
            #test_df['neg_samples'] = test_df['SessionId'].apply(lambda x: eval_sessions_neg_samples[x])
            test_df['neg_samples'] = test_df['SessionId'].apply(lambda x: eval_sessions_neg_samples[str(x)] if str(x) in eval_sessions_neg_samples else None)
            test_df = test_df[pd.notnull(test_df['neg_samples'])]

            #TODO: Validate evaluation on neg samples and merge metrics
            metrics_results = evaluate_sessions_batch_neg_samples(gru, streaming_metrics, test_df, items=None,
                cut_off=ARGS.eval_metrics_top_n, batch_size=ARGS.batch_size, 
                session_key='SessionId', item_key='ItemId', 
                time_key='Time', session_neg_samples_key='neg_samples')

            print(metrics_results)
            print("")

            hit_rates.append(metrics_results['hitrate_at_n'])
            mrrs.append(metrics_results['mrr_at_n'])

            eval_sessions_metrics_log.append({'hitrate_at_n_gru4rec': metrics_results['hitrate_at_n'],
                                              'mrr_at_n_gru4rec': metrics_results['mrr_at_n'],
                                              'clicks_count': len(test_df),
                                              'sessions_count': test_df['SessionId'].nunique()})
            save_eval_benchmark_metrics_csv(eval_sessions_metrics_log, temp_folder,
                                            training_hours_for_each_eval=ARGS.training_hours_for_each_eval,
                                            output_csv=EVAL_METRICS_FILE)

    finally:
        print("AVG HitRate: {}".format(sum(hit_rates) / len(hit_rates)))
        print("AVG MRR: {}".format(sum(mrrs) / len(mrrs)))

        #Export trained model
        gru_file = open(os.path.join(temp_folder, MODEL_FILE), "wb+")
        try:
            pkl.dump( gru, gru_file )
        finally:
            gru_file.close()


    print('Trained model and eval results exported to temporary folder: {}'.format(temp_folder))    