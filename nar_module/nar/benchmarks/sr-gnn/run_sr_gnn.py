import os
import argparse
import numpy as np
import pandas as pd
import tempfile

from .gnn_ml_fast import GGNN

from ..benchmarks_data_loader import DataLoader, load_eval_negative_samples

from ...utils import resolve_files, chunks, str2bool
from ...metrics import HitRate, MRR, ItemCoverage, ExpectedRankSensitiveNovelty, ExpectedRankRelevanceSensitiveNovelty, ContentExpectedRankSensitiveIntraListDiversity, ContentExpectedRankRelativeSensitiveIntraListDiversity, ContentExpectedRankRelativeRelevanceSensitiveIntraListDiversity
from ...nar_utils import load_acr_module_resources, save_eval_benchmark_metrics_csv
from ...clicked_items_state import ClickedItemsState


EVAL_METRICS_FILE = 'eval_stats_sr-gnn.csv'

#Configuring Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", type=str, help="'g1' or 'adressa'", default='')
parser.add_argument("--train_set_path_regex", type=str, help="", default='')
parser.add_argument("--eval_sessions_negative_samples_json_path", type=str, help="", default='')
parser.add_argument("--acr_module_resources_path", type=str, help="ACR module resources path", default='')
parser.add_argument("--training_hours_for_each_eval", type=int, help="", default=1)
parser.add_argument("--eval_metrics_top_n", type=int, help="", default=10)
parser.add_argument("--batch_size", type=int, help="", default=128)
parser.add_argument("--n_epochs", type=int, help="", default=3)
parser.add_argument("--hidden_size", type=int, help='hidden state size', default=100)
parser.add_argument("--l2_lambda", type=float, help="l2 penalty", default=1e-5)
parser.add_argument("--propagation_steps", type=int, help="gnn propagation steps", default=1)
parser.add_argument("--learning_rate", type=float, help="learning rate", default=0.001)
parser.add_argument("--learning_rate_decay", type=float, help="learning rate decay rate", default=0.1)
parser.add_argument("--learning_rate_decay_steps", type=int, help="the number of steps after which the learning rate decays", default=3)
parser.add_argument('--nonhybrid', type=str2bool, nargs='?', const=True, default=True, help='global preference')
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

        #print('unique_items',train_df['ItemId'].nunique())

        #Using the first hour of next training chunck as eval
        eval_file = training_files_chunks[chunk_id+1][0]

        print('Evaluation file {}'.format(eval_file))        
        test_df = data_loader.load_dataframe(eval_file)

        #Checking if a session for test set is missing the negative samples
        test_df['neg_samples'] = test_df['SessionId'].apply(lambda x: eval_sessions_neg_samples[x][0] if x in eval_sessions_neg_samples else None)
        #print('test_df', len(test_df))        
        assert len(test_df[pd.isnull(test_df['neg_samples'])]) == 0, "There are sessions without negative samples"
        #print('test_df WITHOUT NEG SAMPLES', len(test_df))

        if len(test_df) == 0:
            print('WARNING: Ignoring test file for having no clicks')
            eval_sessions_metrics_log.append({
                                          'clicks_count': 0,
                                          'sessions_count': 0})
            print('')
            
            continue

        #print(test_df['neg_samples'].head(5))            


        model = GGNN(hidden_size=ARGS.hidden_size, out_size=ARGS.hidden_size, batch_size=ARGS.batch_size, 
             l2=ARGS.l2_lambda,  step=ARGS.propagation_steps, 
             lr=ARGS.learning_rate, lr_dc=ARGS.learning_rate_decay, lr_dc_step=ARGS.learning_rate_decay_steps, 
             nonhybrid=ARGS.nonhybrid, epoch_n=ARGS.n_epochs)


        
        train_data, test_data, count_clicks_in_test_items_not_in_train_set = model.prepare_data(train_df, test_df, eval_sessions_neg_samples)

        print('Starting training')
        model.fit(train_data)


        #Updating recent clicks and popularity info with all clicks in the train set
        clicked_items_state.update_items_state(train_df['ItemId'].values, train_df['Time'].values) 
        #print('buffer.N', np.count_nonzero(clicked_items_state.get_recent_clicks_buffer()))
        min_timestamp_testset = np.min(test_df['Time'].values)
        clicked_items_state.truncate_last_hours_recent_clicks_buffer(min_timestamp_testset)
        #print('buffer.N.AFTER_TRUNC', np.count_nonzero(clicked_items_state.get_recent_clicks_buffer()))


        #Save state of items popularity and recency from train loop, to restore after evaluation finishes
        clicked_items_state.save_state_checkpoint() 

        print('Starting evaluation')
        #Setup metrics
        streaming_metrics = create_eval_metrics(ARGS.eval_metrics_top_n, 
                                                ARGS.eval_negative_sample_relevance,
                                                content_article_embeddings_matrix,
                                                clicked_items_state)

        metric_results = model.evaluate(test_data, streaming_metrics, clicked_items_state, count_clicks_in_test_items_not_in_train_set, min_timestamp_testset)

        #Restoring the original state of items popularity and recency state from train loop
        clicked_items_state.restore_state_checkpoint()

        #metric_results = model.fit_and_evaluate(train_df, test=test_df, eval_sessions_neg_samples=eval_sessions_neg_samples, 
        #    eval_top_k=10, streaming_metrics=streaming_metrics)

        log_eval_metrics(metric_results, eval_sessions_metrics_log, test_df, temp_folder)

        print('')

    print('Job finished')