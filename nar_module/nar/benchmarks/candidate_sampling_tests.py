from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

from .candidate_sampling import CandidateSamplingManager

class CandidateSamplingManagerTestCase(unittest.TestCase):

    def setUp(self):
        pop_recent_items_buffer = np.array([1,2,3,1,2,3,4,4,4,5,5,5,6,6,7,7,8,9,10,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.get_recent_clicks_buffer_fn = lambda: pop_recent_items_buffer
        self.sampling_manager = CandidateSamplingManager(self.get_recent_clicks_buffer_fn)

    def test_get_sample_from_recently_clicked_items_buffer(self):
        sample = self.sampling_manager.get_sample_from_recently_clicked_items_buffer(5)
        self.assertEqual(sample.shape, (5,), "Should return 5 samples")
        self.assertNotIn(0, sample, "Should not contain 0 values")

    def test_get_neg_items_click(self):
        valid_samples_session=[1,2,2,4,4,5,4,3,2,16,4,8,6]
        sample = self.sampling_manager.get_neg_items_click(valid_samples_session, num_neg_samples=5)
        self.assertEqual(sample.shape, (5,), "Should return 5 samples")
        self.assertEqual(np.unique(sample).shape, (5,), "Should return 5 unique samples")

    def test_get_neg_items_click_padding(self):
        valid_samples_session=[1,2,2]
        sample = self.sampling_manager.get_neg_items_click(valid_samples_session, num_neg_samples=10)
        self.assertEqual(sample.shape, (10,), "Should return 10 samples")
        self.assertEqual(np.count_nonzero(sample), 2, "Should return 2 unique non-zeroed samples" )
        self.assertFalse(sample[2:].any(), "Should return 8 last items zeroed")


    def test_get_neg_items_session(self):
        session_item_ids = [1,2,3]
        candidate_samples = [1,3,5,7,9,11,13,15,18,20,9,11]
        num_neg_samples=10
        samples = self.sampling_manager.get_neg_items_session(session_item_ids, candidate_samples, num_neg_samples)
        self.assertEqual(samples.shape, (3,10), "Should return 10 samples")
        self.assertEqual(np.count_nonzero(samples==0), 2*3, "Should return 6 zeroed samples" )
        self.assertFalse(samples[:,-2:].any(), "Should have last 2 columns zeroed")
        for i in session_item_ids:
            self.assertNotIn(i, samples, "Should not contain clicked items among negative samples")


    def test_get_neg_items_session_not_ignore_session_items(self):
        session_item_ids = [1,2,3]
        candidate_samples = [1,3,5,7,9,2,13,15,18,20, 9]
        num_neg_samples=10
        new_sampling_manager = CandidateSamplingManager(self.get_recent_clicks_buffer_fn, 
                                                        ignore_session_items_on_sampling=False)
        samples = new_sampling_manager.get_neg_items_session(session_item_ids, candidate_samples, num_neg_samples)
        self.assertEqual(samples.shape, (3,10), "Should return 10 samples")

        for i in session_item_ids:
            self.assertIn(i, samples, "Should contain clicked items among candidate samples")


    def test_get_negative_samples(self):
        sessions_item_ids = np.array([[1,2,3],
                                      [4,0,0]])
        candidate_samples = [1,3,5,7,9,11,13,15,18,20,9,11]
        num_neg_samples=10
        samples = self.sampling_manager.get_negative_samples(sessions_item_ids, candidate_samples, num_neg_samples)
        self.assertEqual(samples.shape, (2, 3, 10), "Should return 2 sessions with at most 3 clicks and up to 10 neg. samples")
        self.assertEqual(np.count_nonzero(samples==0), 2*10 + 2*3, "Should return 6 zeroed samples" )
        self.assertFalse(samples[1,-2:].any(), "Should have last 2 rows zeroed")
        for session, neg_samples in zip(sessions_item_ids, samples):
            self.assertTrue(len(set(session.ravel()).intersection(set(neg_samples.ravel())).difference({0})) == 0, "Should not contain clicked items among negative samples")


    def test_get_batch_negative_samples_by_session(self):
        sessions_item_ids = np.array([[1,2,3,4,5],
                                      [4,5,6,7,0]])
        candidate_samples = [2,2,3,3,4,4,4,5,5,5,5,6,7,10,10,10,11,11,12,12,13,15]
        num_neg_samples=3
        samples = self.sampling_manager.get_batch_negative_samples_by_session(sessions_item_ids, candidate_samples, 
                                                                              num_negative_samples=num_neg_samples, 
                                                                              first_sampling_multiplying_factor=2)

        self.assertEqual(samples.shape, (2, 5, 3), "Should return 2 sessions with at most 5 clicks and up to 3 neg. samples")
        self.assertFalse(samples[1,-1].any(), "Should have last session with no neg sample")
        for session, neg_samples in zip(sessions_item_ids, samples):
            self.assertTrue(len(set(session.ravel()).intersection(set(neg_samples.ravel())).difference({0})) == 0, "Should not contain clicked items among negative samples")


    def test_get_batch_negative_samples(self):
        sessions_item_ids = np.array([[1,2,3,4,5],
                                      [4,5,6,7,0]])
        negative_samples_by_session=4
        negative_sample_from_buffer=10
        samples = self.sampling_manager.get_batch_negative_samples(sessions_item_ids, 
                                    negative_samples_by_session, negative_sample_from_buffer)

        self.assertEqual(samples.shape, (2, 5, 4), "Should return 2 sessions with at most 5 clicks and up to 4 neg. samples")
        self.assertFalse(samples[1,-1].any(), "Should have last session with no neg sample")
        for session, neg_samples in zip(sessions_item_ids, samples):
            self.assertTrue(len(set(session.ravel()).intersection(set(neg_samples.ravel())).difference({0})) == 0, "Should not contain clicked items among negative samples")

if __name__ == '__main__':
    unittest.main()