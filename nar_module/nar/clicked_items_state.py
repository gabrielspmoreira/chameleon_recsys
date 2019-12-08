import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import Counter
from copy import deepcopy
from itertools import permutations

from .evaluation import ColdStartAnalysisState

class ClickedItemsState:
    
    def __init__(self, recent_clicks_buffer_hours, recent_clicks_buffer_max_size, recent_clicks_for_normalization, num_items):
        self.recent_clicks_buffer_hours = recent_clicks_buffer_hours
        self.recent_clicks_buffer_max_size = recent_clicks_buffer_max_size
        self.recent_clicks_for_normalization = recent_clicks_for_normalization
        self.num_items = num_items      
        self.reset_state()
        
    def reset_state(self):
        #Global state
        self.articles_pop = np.zeros(shape=[self.num_items], dtype=np.int64)    
            
        self.articles_recent_pop = np.zeros(shape=[self.num_items], dtype=np.int64)
        self._update_recent_pop_norm(self.articles_recent_pop)

        #Clicked buffer has two columns (article_id, click_timestamp)
        self.pop_recent_clicks_buffer = np.zeros(shape=[self.recent_clicks_buffer_max_size, 2], dtype=np.int64)
        self.pop_recent_buffer_article_id_column = 0
        self.pop_recent_buffer_timestamp_column = 1


        #State shared by ItemCooccurrenceRecommender and ItemKNNRecommender
        self.items_coocurrences = csr_matrix((self.num_items, self.num_items), dtype=np.int64) 

        #States specific for benchmarks
        self.benchmarks_states = dict()

        #Stores the timestamp of the first click in the item
        self.items_first_click_ts = dict()
        #Stores the delay (in minutes) from item's first click to item's first recommendation from CHAMELEON
        self.items_delay_for_first_recommendation = dict()

        self.current_step = 0
        self.items_first_click_step = dict()
        self.cold_start_state = ColdStartAnalysisState()


        
    def save_state_checkpoint(self):
        self.articles_pop_chkp = np.copy(self.articles_pop)
        self.pop_recent_clicks_buffer_chkp = np.copy(self.pop_recent_clicks_buffer)
        self.items_coocurrences_chkp = csr_matrix.copy(self.items_coocurrences)        
        self.benchmarks_states_chkp = deepcopy(self.benchmarks_states)
        self.items_first_click_ts_chkp = deepcopy(self.items_first_click_ts)
        self.items_delay_for_first_recommendation_chkp = deepcopy(self.items_delay_for_first_recommendation)  
        
        self.items_first_click_step_chkp = deepcopy(self.items_first_click_step) 
        self.cold_start_state_chkp = deepcopy(self.cold_start_state)
        self.current_step_chkp = self.current_step
        
    def restore_state_checkpoint(self):
        self.articles_pop = self.articles_pop_chkp
        del self.articles_pop_chkp
        self.pop_recent_clicks_buffer = self.pop_recent_clicks_buffer_chkp
        del self.pop_recent_clicks_buffer_chkp
        self.items_coocurrences = self.items_coocurrences_chkp
        del self.items_coocurrences_chkp
        self.items_first_click_ts = self.items_first_click_ts_chkp
        del self.items_first_click_ts_chkp
        self.items_delay_for_first_recommendation = self.items_delay_for_first_recommendation_chkp
        del self.items_delay_for_first_recommendation_chkp
        self.benchmarks_states = self.benchmarks_states_chkp
        del self.benchmarks_states_chkp
        
        self.items_first_click_step = self.items_first_click_step_chkp
        del self.items_first_click_step_chkp
        self.cold_start_state = self.cold_start_state_chkp
        del self.cold_start_state_chkp
        self.current_step = self.current_step_chkp
        
    def get_articles_pop(self):
        return self.articles_pop

    def get_articles_recent_pop(self):
        return self.articles_recent_pop

    def get_articles_recent_pop_norm(self):
        return self.articles_recent_pop_norm
    
    def get_recent_clicks_buffer(self):
        #Returns only the first column (article_id)
        return self.pop_recent_clicks_buffer[:,self.pop_recent_buffer_article_id_column]
    
    def get_items_coocurrences(self):
        return self.items_coocurrences

    def increment_current_step(self):
        self.current_step += 1        

    def get_current_step(self):
        return self.current_step

    def get_cold_start_state(self):
        return self.cold_start_state


    def get_max_timestamp_recent_clicks(self):
        return np.max(self.pop_recent_clicks_buffer[:,self.pop_recent_buffer_timestamp_column])


    def update_items_first_click_ts(self, batch_clicked_items, batch_clicked_timestamps):

        batch_item_ids = batch_clicked_items.reshape(-1)
        batch_clicks_timestamp = batch_clicked_timestamps.reshape(-1)
        sorted_item_clicks = sorted(zip(batch_clicks_timestamp, batch_item_ids))

        for click_ts, item_id in sorted_item_clicks:
            if item_id != 0 and click_ts == 0:
                tf.logging.warn('Item {} has timestamp {}. Original clicked_items: {}. Original timestamps: {}'.format(item_id, click_ts, batch_clicked_items, batch_clicked_timestamps))
            #Ignoring padded items
            elif item_id != 0 and (not item_id in self.items_first_click_ts or click_ts < self.items_first_click_ts[item_id]):
                self.items_first_click_ts[item_id] = click_ts

    '''
    def update_items_delay_for_first_recommendation(self, batch_rec_items, batch_click_timestamps, topn):
        batch_top_rec_ids = batch_rec_items[:,:,:topn]

        #Repeating last dimension of click timestamp to the number of recommendations, to make matrices compatible
        batch_rec_timestamp = np.tile(batch_click_timestamps, (1,1,topn))

        batch_top_rec_ids = batch_top_rec_ids.reshape(-1)
        batch_rec_timestamp = batch_rec_timestamp.reshape(-1)

        sorted_item_recs = list(sorted(zip(batch_rec_timestamp, batch_top_rec_ids)))

        neg_delay = 0
        valid_delay = 0
        for rec_ts, item_id in sorted_item_recs:
            #Ignoring padded items
            if rec_ts != 0 and item_id != 0:
                if item_id in self.items_first_click_ts:
                    delay_minutes = (rec_ts - self.items_first_click_ts[item_id]) / (1000. * 60.)

                    if delay_minutes > 0 and \
                        (not item_id in self.items_delay_for_first_recommendation or \
                         delay_minutes < self.items_delay_for_first_recommendation[item_id]):

                        #tf.logging.info('rec_ts: {}, items_first_click_ts: {}, delay: {}'.format(rec_ts, self.items_first_click_ts[item_id], delay))
                        self.items_delay_for_first_recommendation[item_id] = delay_minutes
                #else:
                #    tf.logging.warn('Item {} not found in clicked items'.format(item_id))


    def log_stats_time_for_first_rec(self):
        #tf.logging.info('log_stats_time_for_first_rec: {}'.format(len(self.items_delay_for_first_recommendation)))
        if len(self.items_delay_for_first_recommendation) > 0:
            values = np.array(list(self.items_delay_for_first_recommendation.values()))
            stats = {'min': np.min(values),
                    '10%': np.percentile(values, 10),
                    '25%': np.percentile(values, 25),
                    '50%': np.percentile(values, 50),
                    '75%': np.percentile(values, 75),
                    '90%': np.percentile(values, 90),
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                     }

            tf.logging.info('Stats on delay for first recommendation since first click: {}'.format(stats))


            #Crossing popularity with time for first rec
            items_pop_time_for_first_rec = []
            for item_id in self.items_delay_for_first_recommendation.keys():
                time_for_first_rec = self.items_delay_for_first_recommendation[item_id]
                item_pop = self.articles_pop[item_id]
                items_pop_time_for_first_rec.append((item_pop, time_for_first_rec))

            items_pop_time_for_first_rec_df = pd.DataFrame(items_pop_time_for_first_rec, columns=['pop', 'time_to_rec'])
            #Binning popularity
            items_pop_time_for_first_rec_df['pop_deciles_binned'] = pd.qcut(items_pop_time_for_first_rec_df['pop'], 10, duplicates='drop')
            time_to_rec_by_popularity_df = items_pop_time_for_first_rec_df.groupby('pop_deciles_binned')['time_to_rec'].agg(['median', 'mean', 'std'])

            tf.logging.info('Stats on delay for first recommendation since first click (BY POPULARITY): {}'.format(time_to_rec_by_popularity_df))
    '''

    def update_items_state(self, batch_clicked_items, batch_clicked_timestamps):
        #batch_items_nonzero = self._get_non_zero_items_vector(batch_clicked_items)

        self._update_recently_clicked_items_buffer(batch_clicked_items, batch_clicked_timestamps)
        self._update_recent_pop_items()

        self._update_pop_items(batch_clicked_items)   


    def update_items_first_click_step(self, batch_clicked_items):        
        
        #Getting the unique clicked items in the batch
        batch_clicked_items_set = set(batch_clicked_items).difference(set([0]))

        for item_id in batch_clicked_items_set:
            if item_id not in self.items_first_click_step:
                self.items_first_click_step[item_id] = self.get_current_step()

    
    def _update_recently_clicked_items_buffer(self, batch_clicked_items, batch_clicked_timestamps):

        #Concatenating column vectors of batch clicked items
        batch_recent_clicks_timestamps = np.hstack([batch_clicked_items.reshape(-1,1), batch_clicked_timestamps.reshape(-1,1)])
        #Inverting the order of clicks, so that latter clicks are now the first in the vector
        batch_recent_clicks_timestamps = batch_recent_clicks_timestamps[::-1]
        
        #Keeping in the buffer only clicks within the last N hours
        min_timestamp_batch = np.min(batch_clicked_timestamps)

        self.truncate_last_hours_recent_clicks_buffer(min_timestamp_batch)
        
        #Concatenating batch clicks with recent buffer clicks, limited by the buffer size
        self.pop_recent_clicks_buffer = np.vstack([batch_recent_clicks_timestamps, self.pop_recent_clicks_buffer])[:self.recent_clicks_buffer_max_size]
        #Complete buffer with zeroes if necessary
        if self.pop_recent_clicks_buffer.shape[0] < self.recent_clicks_buffer_max_size:
            self.pop_recent_clicks_buffer = np.vstack([self.pop_recent_clicks_buffer, 
                                                       np.zeros(shape=[self.recent_clicks_buffer_max_size-self.pop_recent_clicks_buffer.shape[0], 2], dtype=np.int64)])
        
    def truncate_last_hours_recent_clicks_buffer(self, reference_timestamp):
        MILISECS_BY_HOUR = 1000 * 60 * 60     
        min_timestamp_buffer_threshold = reference_timestamp - int(self.recent_clicks_buffer_hours * MILISECS_BY_HOUR)
        self.pop_recent_clicks_buffer = self.pop_recent_clicks_buffer[self.pop_recent_clicks_buffer[:,self.pop_recent_buffer_timestamp_column]>=min_timestamp_buffer_threshold]


    def _update_recent_pop_items(self):
        #Using all the buffer to compute items popularity
        pop_recent_clicks_buffer_items = self.pop_recent_clicks_buffer[:, self.pop_recent_buffer_article_id_column]
        recent_clicks_buffer_nonzero = pop_recent_clicks_buffer_items[np.nonzero(pop_recent_clicks_buffer_items)]
        recent_clicks_item_counter = Counter(recent_clicks_buffer_nonzero)
        
        self.articles_recent_pop = np.zeros(shape=[self.num_items], dtype=np.int64)
        self.articles_recent_pop[list(recent_clicks_item_counter.keys())] = list(recent_clicks_item_counter.values())

        self._update_recent_pop_norm(self.articles_recent_pop)

    def _update_recent_pop_norm(self, articles_recent_pop):
        #Minimum value for norm_pop, to avoid 0
        min_norm_pop = 1.0/self.recent_clicks_for_normalization
        self.articles_recent_pop_norm = np.maximum(articles_recent_pop / (articles_recent_pop.sum() + 1), 
                                                   [min_norm_pop])

    def _update_pop_items(self, batch_items_nonzero):
        batch_item_counter = Counter(batch_items_nonzero)
        self.articles_pop[list(batch_item_counter.keys())] += list(batch_item_counter.values())
        
    def update_items_coocurrences(self, batch_clicked_items):
        for session_items in batch_clicked_items:
            session_pairs = permutations(session_items[np.nonzero(session_items)], r=2)
            rows, cols = zip(*session_pairs)
            self.items_coocurrences[rows, cols] += 1