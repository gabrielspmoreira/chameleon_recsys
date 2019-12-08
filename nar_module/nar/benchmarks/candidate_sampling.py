from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class CandidateSamplingManager():

    def __init__(self, get_recent_clicks_buffer_fn, ignore_session_items_on_sampling=True):
        self.get_recent_clicks_buffer_fn = get_recent_clicks_buffer_fn
        self.ignore_session_items_on_sampling = ignore_session_items_on_sampling

    def get_sample_from_recently_clicked_items_buffer(self, sample_size):
        pop_recent_items_buffer = self.get_recent_clicks_buffer_fn()
        pop_recent_items_buffer_masked = pop_recent_items_buffer.ravel()[np.flatnonzero(pop_recent_items_buffer)]
                
        pop_recent_items_buffer_shuffled = np.random.permutation(pop_recent_items_buffer_masked)
        
        #Samples K articles from recent clicks (popularity sampled)
        sample_recently_clicked_items = pop_recent_items_buffer_shuffled[:sample_size]
        return sample_recently_clicked_items



    def get_neg_items_click(self, valid_samples_session, num_neg_samples):
        #Shuffles neg. samples for each click
        valid_samples_shuffled = np.random.permutation(valid_samples_session)
        
        samples_unique_vals, samples_unique_idx = np.unique(valid_samples_shuffled, return_index=True)

        #Returning first N unique items (to avoid repetition)
        first_unique_items = samples_unique_vals[np.argsort(samples_unique_idx)][:num_neg_samples]

        #Padding if necessary to keep the number of neg samples constant (ex: first batch)
        first_unique_items_padded_if_needed = np.concatenate([first_unique_items, np.zeros(num_neg_samples-first_unique_items.shape[0], np.int64)], axis=0)

        return first_unique_items_padded_if_needed                            


    def get_neg_items_session(self, session_item_ids, candidate_samples, num_neg_samples):
        if self.ignore_session_items_on_sampling:
            #Ignoring negative samples clicked within the session (keeps the order and repetition of candidate_samples)
            samples_for_session = np.setdiff1d(candidate_samples, session_item_ids, assume_unique=True)
        else:
            samples_for_session = candidate_samples

        #Generating a random list of negative samples for each click (with no repetition)
        session_clicks_neg_items = np.vstack([self.get_neg_items_click(samples_for_session, num_neg_samples) \
                                              if click_id != 0 \
                                              else np.zeros(num_neg_samples, np.int64) \
                                              for click_id in session_item_ids])

        return session_clicks_neg_items
        

    def get_negative_samples(self, all_clicked_items, candidate_samples, num_neg_samples): 
        #Shuffling negative samples by session and limiting to num_neg_samples 
        return np.vstack([np.expand_dims(self.get_neg_items_session(session_item_ids, candidate_samples, num_neg_samples), 0) \
                          for session_item_ids in all_clicked_items])

         
    def get_batch_negative_samples_by_session(self, all_clicked_items, additional_samples, num_negative_samples, 
                                         first_sampling_multiplying_factor=20):
        batch_items = all_clicked_items.ravel()
        
        #Removing padded (zeroed) items
        batch_items_non_zero = batch_items[np.flatnonzero(batch_items)]           
                        
        #Concatenating batch items with additional samples (to deal with small batches)
        candidate_neg_items = np.concatenate([batch_items_non_zero, additional_samples], axis=0)     

        #Shuffling candidates and sampling the first 20N (1000 if neg_samples=50)
        candidate_neg_items_shuffled = np.random.permutation(candidate_neg_items)
        candidate_neg_items_sampled = candidate_neg_items_shuffled[:(num_negative_samples*first_sampling_multiplying_factor)]

        batch_negative_items = self.get_negative_samples(all_clicked_items, candidate_neg_items_sampled, num_negative_samples) 
        
        return batch_negative_items


    def get_batch_negative_samples(self, all_clicked_items, negative_samples_by_session, negative_sample_from_buffer):
        #Samples from recent items buffer
        negative_sample_recently_clicked_ids = self.get_sample_from_recently_clicked_items_buffer(
                                                            negative_sample_from_buffer)            

        
        batch_negative_items = self.get_batch_negative_samples_by_session(all_clicked_items, 
                                                        additional_samples=negative_sample_recently_clicked_ids, 
                                                        num_negative_samples=negative_samples_by_session)

        return batch_negative_items