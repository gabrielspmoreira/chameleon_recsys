import os
import argparse
import numpy as np
import pandas as pd
import tempfile


from .gru4rec2 import GRU4Rec
from .gru4rec2_evaluation import evaluate_sessions_batch_neg_samples
from ..benchmarks_data_loader import DataLoader, load_eval_negative_samples

from ...datasets import make_dataset
from ...utils import resolve_files, chunks
from ...metrics import HitRate, MRR, ItemCoverage, ExpectedRankSensitiveNovelty, ExpectedRankRelevanceSensitiveNovelty, ContentExpectedRankSensitiveIntraListDiversity, ContentExpectedRankRelativeSensitiveIntraListDiversity, ContentExpectedRankRelativeRelevanceSensitiveIntraListDiversity
from ...nar_utils import load_acr_module_resources, save_eval_benchmark_metrics_csv
from ...clicked_items_state import ClickedItemsState

import theano.misc.pkl_utils as pkl



#Configuring Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", type=str, help="'g1' or 'adressa'", default='')
parser.add_argument("--train_set_path_regex", type=str, help="", default='')
parser.add_argument("--eval_sessions_negative_samples_json_path", type=str, help="", default='')
parser.add_argument("--acr_module_resources_path", type=str, help="ACR module resources path", default='')
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
#Params for recent clicks buffer and popularity
parser.add_argument('--recent_clicks_buffer_hours', type=float, default=1.0, help='Number of hours that will be kept in the recent clicks buffer (limited by recent_clicks_buffer_max_size)')
parser.add_argument('--recent_clicks_buffer_max_size', type=int, default=20000, help='Maximum size of recent clicks buffer')
parser.add_argument('--recent_clicks_for_normalization', type=int, default=2000, help='Number of recent clicks to consider to normalize recency and populary (and novelty) dynamic features')
parser.add_argument('--eval_negative_sample_relevance', type=float, default=0.02, help='Relevance of negative samples within top-n recommended items for evaluation (relevance of positive sample is always 1.0)')
ARGS = parser.parse_args()

#Disabling TF logs in console
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.logging.set_verbosity(tf.logging.ERROR)


MODEL_FILE = "gru_model.mdl"
EVAL_METRICS_FILE = 'eval_stats_gru4rec_{}.csv'.format(ARGS.dataset_type)


def log_eval_metrics(metrics_results, eval_sessions_metrics_log, test_df, temp_folder):
    metrics_results['clicks_count'] = len(test_df)
    metrics_results['sessions_count'] = test_df['SessionId'].nunique()

    eval_sessions_metrics_log.append(metrics_results)

    print('Exporting results to temporary folder: {}'.format(temp_folder))    
    save_eval_benchmark_metrics_csv(eval_sessions_metrics_log, temp_folder,
                                    training_hours_for_each_eval=ARGS.training_hours_for_each_eval,
                                    output_csv=EVAL_METRICS_FILE)


def create_eval_metrics(top_n, 
                        eval_negative_sample_relevance,
                        content_article_embeddings_matrix,
                        clicked_items_state):

    relevance_positive_sample = 1.0
    #Empirical: The weight of negative samples
    relevance_negative_samples = eval_negative_sample_relevance     

    recent_clicks_buffer = clicked_items_state.get_recent_clicks_buffer()    

    eval_metrics = [metric(topn=top_n) for metric in [HitRate, MRR]]

    #TODO: Known issue: Item coverage here is not considering as recommendable items those who were not in the train set
    eval_metrics.append(ItemCoverage(top_n, recent_clicks_buffer))  
    eval_metrics.append(ExpectedRankSensitiveNovelty(top_n)) 
    eval_metrics.append(ExpectedRankRelevanceSensitiveNovelty(top_n, relevance_positive_sample, relevance_negative_samples))
    eval_metrics.append(ContentExpectedRankRelativeSensitiveIntraListDiversity(top_n, content_article_embeddings_matrix))        
    eval_metrics.append(ContentExpectedRankRelativeRelevanceSensitiveIntraListDiversity(top_n, content_article_embeddings_matrix, relevance_positive_sample, relevance_negative_samples))

    return eval_metrics


if __name__ == '__main__':

    temp_folder = tempfile.mkdtemp()
    print('Creating temporary folder: {}'.format(temp_folder))


    
    '''
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
    '''

    try:

        eval_sessions_neg_samples = load_eval_negative_samples(ARGS.eval_sessions_negative_samples_json_path)

        train_files = resolve_files(ARGS.train_set_path_regex)
        training_files_chunks = list(chunks(train_files, ARGS.training_hours_for_each_eval))
        
        data_loader = DataLoader(ARGS.dataset_type)

        print('Loading ACR module assets')
        _, _, content_article_embeddings_matrix = \
                load_acr_module_resources(ARGS.acr_module_resources_path)


        clicked_items_state = ClickedItemsState(ARGS.recent_clicks_buffer_hours,
                                                ARGS.recent_clicks_buffer_max_size, 
                                                ARGS.recent_clicks_for_normalization, 
                                                content_article_embeddings_matrix.shape[0])

        eval_sessions_metrics_log = []

        print("Starting Training Loop")
        for chunk_id in range(0, len(training_files_chunks)-1):
            
            training_files_chunk = training_files_chunks[chunk_id]
            print('Training from file {} to {}'.format(training_files_chunk[0], training_files_chunk[-1]))
            train_df = data_loader.load_dataframe(training_files_chunk)

            if train_df['SessionId'].nunique() < ARGS.batch_size:
                print('WARNING: Ignoring training file for having less than {} sessions: {}'.format(ARGS.batch_size, train_df['SessionId'].nunique()))
                print('')
                continue

            gru = GRU4Rec(n_epochs=ARGS.n_epochs, loss='bpr-max-0.5', final_act='linear', 
                        hidden_act='tanh', layers=[300], adapt=ARGS.optimizer, decay=0.0, 
                        batch_size=ARGS.batch_size, 
                        dropout_p_embed=0.0, dropout_p_hidden=ARGS.dropout_p_hidden, 
                        learning_rate=ARGS.learning_rate, 
                        lmbd=ARGS.l2_lambda, momentum=ARGS.momentum, n_sample=200, 
                        sample_alpha=0.0, time_sort=False, embedding=ARGS.embedding, #grad_cap=2.0,
                        session_key='SessionId', item_key='ItemId', time_key='Time',
                        clicked_items_state=clicked_items_state)

            #gru.fit(train_df, retrain=hasattr(gru, 'n_items'), sample_store=0)
            

            gru.fit(train_df, retrain=False, sample_store=0)

            items_inverted_dict = gru.items_inverted_dict


            
            ##################### Running 2 additional epochs with last hour file #####################
            
            last_train_file = training_files_chunks[chunk_id][-1]
            print('Running 2 additional epochs with last hour file: {}'.format(last_train_file))
            
            train_df = data_loader.load_dataframe(last_train_file)

            if train_df['SessionId'].nunique() < ARGS.batch_size:
                print('WARNING: Ignoring training file for having less than {} sessions: {}'.format(ARGS.batch_size, train_df['SessionId'].nunique()))
                print('')
                continue



            for _ in range(0,2):
                #gru.fit(train_df, retrain=hasattr(gru, 'n_items'), sample_store=0)
                gru.fit(train_df, retrain=True, sample_store=0)
            
            ####################################################################################
            
            
            


            #Using the first hour of next training chunck as eval
            eval_file = training_files_chunks[chunk_id+1][0]

            print('Evaluating file {}'.format(eval_file))
            test_df = data_loader.load_dataframe(eval_file)
            test_set_count = len(test_df)

            #IMPORTANT: Filtering for prediction only items present in train set, to avoid errors on GRU4REC
            test_df = pd.merge(test_df, pd.DataFrame({'ItemIdx': gru.itemidmap.values, 
                                                      'ItemId': gru.itemidmap.index}), on='ItemId', how='inner')

            count_clicks_in_test_items_not_in_train_set = test_set_count - len(test_df)

            if count_clicks_in_test_items_not_in_train_set > 0:
                perc_test_items_not_found = count_clicks_in_test_items_not_in_train_set / test_set_count
                print('{} ({}%) test set clicks in items not present in train set.'.format(count_clicks_in_test_items_not_in_train_set, perc_test_items_not_found))


            test_df = test_df.groupby('SessionId').filter(lambda x: len(x) > 1)

            


            if len(test_df) == 0:
                print('WARNING: Ignoring test file for having no clicks')
                eval_sessions_metrics_log.append({'clicks_count': 0,
                                              'sessions_count': 0})
                print('')
                
                continue

            elif test_df['SessionId'].nunique() < ARGS.batch_size:
                print('WARNING: Ignoring test file for having less than {} sessions: {}'.format(ARGS.batch_size, test_df['SessionId'].nunique()))
                eval_sessions_metrics_log.append({'clicks_count': len(test_df),
                                              'sessions_count': test_df['SessionId'].nunique()})
                print('')

                continue

              
            #test_df['neg_samples'] = test_df['SessionId'].apply(lambda x: eval_sessions_neg_samples[x])
            test_df['neg_samples'] = test_df['SessionId'].apply(lambda x: eval_sessions_neg_samples[x][0] if x in eval_sessions_neg_samples else None)
            assert len(test_df[pd.isnull(test_df['neg_samples'])]) == 0, "There are sessions without negative samples"


            #Setup metrics
            streaming_metrics = create_eval_metrics(ARGS.eval_metrics_top_n, 
                                                    ARGS.eval_negative_sample_relevance,
                                                    content_article_embeddings_matrix,
                                                    clicked_items_state)


            #Save state of items popularity and recency from train loop, to restore after evaluation finishes
            clicked_items_state.save_state_checkpoint() 

            #TODO: Validate evaluation on neg samples and merge metrics
            metrics_results = evaluate_sessions_batch_neg_samples(gru, streaming_metrics, test_df, clicked_items_state, items=None,
                cut_off=ARGS.eval_metrics_top_n, batch_size=ARGS.batch_size, 
                count_clicks_in_test_items_not_in_train_set=count_clicks_in_test_items_not_in_train_set,
                items_inverted_dict=items_inverted_dict,
                session_key='SessionId', item_key='ItemId', 
                time_key='Time', session_neg_samples_key='neg_samples')


            print(metrics_results)
            print("")

            #Restoring the original state of items popularity and recency state from train loop
            clicked_items_state.restore_state_checkpoint()

            #metric_results = model.fit_and_evaluate(train_df, test=test_df, eval_sessions_neg_samples=eval_sessions_neg_samples, 
            #    eval_top_k=10, streaming_metrics=streaming_metrics)

            log_eval_metrics(metrics_results, eval_sessions_metrics_log, test_df, temp_folder)

            print('')

            '''
            hit_rates.append(metrics_results['hitrate_at_n'])
            mrrs.append(metrics_results['mrr_at_n'])

            eval_sessions_metrics_log.append({'hitrate_at_n_gru4rec': metrics_results['hitrate_at_n'],
                                              'mrr_at_n_gru4rec': metrics_results['mrr_at_n'],
                                              'clicks_count': len(test_df),
                                              'sessions_count': test_df['SessionId'].nunique()})
            save_eval_benchmark_metrics_csv(eval_sessions_metrics_log, temp_folder,
                                            training_hours_for_each_eval=ARGS.training_hours_for_each_eval,
                                            output_csv=EVAL_METRICS_FILE)
            '''


    finally:
        pass
        '''
        #Export trained model
        gru_file = open(os.path.join(temp_folder, MODEL_FILE), "wb+")
        try:
            pkl.dump( gru, gru_file )
        finally:
            gru_file.close()
        '''


    print('Trained model and eval results exported to temporary folder: {}'.format(temp_folder))    