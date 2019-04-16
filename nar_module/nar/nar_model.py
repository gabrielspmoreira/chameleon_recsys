from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from itertools import permutations
from collections import Counter
from copy import deepcopy
from time import time

from tensorflow.contrib.layers import xavier_initializer, variance_scaling_initializer
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from .metrics import HitRate, HitRateBySessionPosition, MRR, NDCG, ItemCoverage, PopularityBias, CategoryExpectedIntraListDiversity, Novelty, ExpectedRankSensitiveNovelty, ExpectedRankRelevanceSensitiveNovelty, ContentExpectedRankSensitiveIntraListDiversity, ContentExpectedRankRelativeSensitiveIntraListDiversity, ContentExpectedRankRelevanceSensitiveIntraListDiversity, ContentExpectedRankRelativeRelevanceSensitiveIntraListDiversity, ContentAverageIntraListDiversity, ContentMedianIntraListDiversity, ContentMinIntraListDiversity
from .utils import merge_two_dicts, get_tf_dtype, hash_str_to_int, paired_permutations
from .evaluation import update_metrics, compute_metrics_results


ARTICLE_REQ_FEATURES = ['article_id', 'created_at_ts']
SESSION_REQ_SEQ_FEATURES = ['item_clicked', 'event_timestamp']

def get_embedding_size(unique_val_count, const_mult=8):
    return int(math.floor(const_mult * unique_val_count**0.25))

def log_base(x, base):
    numerator = tf.log(tf.to_float(x))
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator

def log_1p(x, base):
    return log_base(x+1, base)

def tf_ndcg_at_k(r, k):
    def _tf_dcg_at_k(r, k):
        last_dim_size = tf.minimum(k, tf.shape(r)[-1])

        input_rank = tf.rank(r)
        input_shape = tf.shape(r)    
        slice_begin = tf.zeros([input_rank], dtype=tf.int32)
        slice_size = tf.concat([input_shape[:-1], [last_dim_size]], axis=0)
        r = tf.slice(tf.to_float(r),
                     begin=slice_begin,
                     size=slice_size)

        last_dim_size = tf.shape(r)[-1]

        dcg = tf.reduce_sum(tf.subtract(tf.pow(2., r), 1) / log_base(tf.range(2, last_dim_size + 2), 2.), axis=-1)

        return dcg    
    
    sorted_values, sorted_idx = tf.nn.top_k(r, k=tf.shape(r)[-1])
    idcg = _tf_dcg_at_k(sorted_values, k)
    
    ndcg = _tf_dcg_at_k(r, k) / idcg
    #Filling up nans (due to zeroed IDCG) with zeros
    ndcg = tf.where(tf.is_nan(ndcg), tf.zeros_like(ndcg), ndcg)

    return ndcg

def cartesian_product(a, b, axis):
    a_rank = tf.rank(a)
    a_dim = tf.shape(a)[axis]    
    b_rank = tf.rank(b)
    b_dim = tf.shape(b)[axis]
    
    axis_a_repeat = tf.sparse_to_dense(sparse_indices=[axis+1], sparse_values=[b_dim], output_shape=[a_rank+1], default_value=1)
    tile_a = tf.tile(tf.expand_dims(a, axis+1), axis_a_repeat)
    
    axis_b_repeat = tf.sparse_to_dense(sparse_indices=[axis], sparse_values=[a_dim], output_shape=[b_rank+1], default_value=1)
    tile_b = tf.tile(tf.expand_dims(b, axis), axis_b_repeat)
    
    cart_prod = tf.concat([tile_a, tile_b], axis=-1)

    #Defining the last dimension of resulting tensor (originally undefined)
    last_dim = int(a.get_shape()[-1]) + int(b.get_shape()[-1])
    cart_prod_shape = list(cart_prod.get_shape())
    cart_prod_shape[-1] = last_dim
    cart_prod.set_shape(cart_prod_shape)    
    
    return cart_prod



def shuffle_columns(x):
    batch_size = tf.shape(x)[0]

    counter = tf.constant(0)
    m0 = tf.zeros(shape=[0, tf.shape(x)[1]], dtype=x.dtype)
    cond = lambda i, m: i < batch_size
    body = lambda i, m: [i+1, tf.concat([m, tf.expand_dims(tf.random_shuffle(x[i]), 0)], axis=0)]
    _, shuffled_columns = tf.while_loop(
        cond, body, loop_vars=[counter, m0],
        shape_invariants=[counter.get_shape(), tf.TensorShape([None,None])])

    return shuffled_columns

class NARModuleModel():
    
    def __init__(self, mode, inputs, labels,  
                 session_features_config,
                 articles_features_config,
                 batch_size, 
                 lr, keep_prob, negative_samples, negative_sample_from_buffer,
                 content_article_embeddings_matrix,
                 rnn_num_layers=1,            
                 softmax_temperature=1.0,
                 reg_weight_decay=0.0, 
                 recent_clicks_buffer_hours=1.0,
                 recent_clicks_buffer_max_size = 1000, 
                 recent_clicks_for_normalization = 1000,
                 articles_metadata=None,
                 plot_histograms=False,
                 metrics_top_n=5,
                 elapsed_days_smooth_log_base=1.3,
                 popularity_smooth_log_base=2.0,
                 CAR_embedding_size=256,
                 rnn_units=256,
                 max_cardinality_for_ohe=10,
                 novelty_reg_factor=0.0,
                 diversity_reg_factor=0.0,
                 internal_features_config={'recency': True,
                                           'novelty': True,
                                           'article_content_embeddings': True,
                                           'item_clicked_embeddings': True}
                ):        
        
        self.lr = lr 
        self.keep_prob = keep_prob
        
        self.elapsed_days_smooth_log_base = elapsed_days_smooth_log_base
        self.popularity_smooth_log_base = popularity_smooth_log_base
        
        self.is_training = (mode == tf.estimator.ModeKeys.TRAIN)   

        self.internal_features_config = internal_features_config
        
        self.negative_samples = negative_samples 
        self.negative_sample_from_buffer = negative_sample_from_buffer

        
        self.rnn_num_layers = rnn_num_layers
        self.metrics_top_n = metrics_top_n
        
        self.plot_histograms = plot_histograms

        self.reg_weight_decay = reg_weight_decay
        self.batch_size = tf.constant(batch_size, dtype=tf.int32)

        self.session_features_config = session_features_config
        self.articles_features_config = articles_features_config

        self.max_cardinality_for_ohe = max_cardinality_for_ohe

        self.novelty_reg_factor = tf.constant(novelty_reg_factor, dtype=tf.float32)
        self.diversity_reg_factor = tf.constant(diversity_reg_factor, dtype=tf.float32)

        self.softmax_temperature = tf.constant(softmax_temperature, dtype=tf.float32, name='softmax_temperature')

        self.recent_clicks_for_normalization = recent_clicks_for_normalization
        

        with tf.variable_scope("article_content_embeddings"):

            self.articles_metadata = {}
            with tf.device('/cpu:0'):
                #Converting Article metadata feature vectors to constants in the graph, to avoid many copies (is saved with the graph)
                for feature_name in articles_metadata:
                    '''
                    self.articles_metadata[feature_name] = tf.constant(articles_metadata[feature_name], 
                                                    shape=articles_metadata[feature_name].shape, 
                                                    dtype=get_tf_dtype(articles_features_config[feature_name]['dtype']))
                    '''
                    self.articles_metadata[feature_name] = tf.placeholder(name="articles_metadata",
                                                    shape=articles_metadata[feature_name].shape, 
                                                    dtype=get_tf_dtype(articles_features_config[feature_name]['dtype']))


            self.items_vocab_size = articles_features_config['article_id']['cardinality']

            #To run on local machine (GPU card with 4 GB RAM), keep Content Article Embeddings constant in CPU memory
            with tf.device('/cpu:0'):

                #Expects vectors within the range [-0.1, 0.1] (min-max scaled) for compatibility with other input features
                self.content_article_embeddings_matrix = tf.placeholder(name="content_article_embeddings_matrix",
                                                               shape=content_article_embeddings_matrix.shape,
                                                               dtype=tf.float32)
        
        with tf.variable_scope("articles_status"):
            with tf.device('/cpu:0'):
                self.articles_recent_pop_norm = tf.placeholder(name="articles_recent_pop_norm",
                                                                shape=[self.items_vocab_size],
                                                                dtype=tf.float32)

            
            self.pop_recent_items_buffer = tf.placeholder(name="pop_recent_items_buffer",
                                               shape=[recent_clicks_buffer_max_size],
                                               dtype=tf.int64)
            tf.summary.scalar('unique_items_clicked_recently', family='stats', tensor=tf.shape(tf.unique(self.pop_recent_items_buffer)[0])[0])   

            tf.summary.scalar('unique_items_clicked_recently_for_normalization', family='stats', tensor=tf.shape(tf.unique(self.pop_recent_items_buffer[:self.recent_clicks_for_normalization])[0])[0])   
       

        #PS: variance_scaling_initializer() is recommended for RELU activations in https://arxiv.org/abs/1502.01852
        #whilst xavier_initializer is recommended for tanh activations
        with tf.variable_scope("main", initializer=xavier_initializer()):
            

            #Initializes CAR item embeddings variable
            #self.create_item_embed_lookup_variable()
            
            
            with tf.variable_scope("inputs"):

                item_clicked = inputs['item_clicked']
                self.item_clicked = item_clicked

                #Control features (ensuring that they keep two dims even when the batch has only one session)
                self.user_id = inputs['user_id']
                self.session_id = inputs['session_id']
                self.session_start = inputs['session_start']

                seq_lengths = inputs['session_size'] - 1 #Ignoring last click only as label
                self.seq_lengths = seq_lengths
                
                #Creates the sessions mask and ensure that rank will be 2 (even when this batch size is 1)
                self.item_clicked_mask = tf.sequence_mask(seq_lengths)
                
                event_timestamp = tf.expand_dims(inputs["event_timestamp"], -1)
                self.event_timestamp = event_timestamp
                max_event_timestamp = tf.reduce_max(event_timestamp)

            
                #Retrieving last label of the sequence
                label_last_item = labels['label_last_item'] 
                self.label_last_item = label_last_item
                all_clicked_items = tf.concat([item_clicked, label_last_item], axis=1)

                #Labels            
                next_item_label = labels['label_next_item']
                self.next_item_label = next_item_label

                batch_max_session_length = tf.shape(next_item_label)[1] 
                batch_current_size = array_ops.shape(next_item_label)[0]

            with tf.variable_scope("batch_stats"):

                #batch_items = self.get_masked_seq_values(inputs['item_clicked']) 
                #Known bug: The article_id 0 will not be considered as negative sample, because padding values also have value 0 
                batch_items_nonzero = tf.boolean_mask(all_clicked_items, tf.cast(tf.sign(all_clicked_items), tf.bool))
                batch_items_count = tf.shape(batch_items_nonzero)[0]
                self.batch_items_count = batch_items_count
                
                batch_unique_items, _ = tf.unique(batch_items_nonzero)
                batch_unique_items_count = tf.shape(batch_unique_items)[0]
                self.batch_unique_items_count = batch_unique_items_count
                
                tf.summary.scalar('batch_items', family='stats', tensor=batch_items_count)
                tf.summary.scalar('batch_unique_items', family='stats', tensor=batch_unique_items_count)
            
            with tf.variable_scope("neg_samples"):
                #Samples from recent items buffer
                negative_sample_recently_clicked_ids = self.get_sample_from_recently_clicked_items_buffer(
                                                                    self.negative_sample_from_buffer)            

                
                batch_negative_items = self.get_batch_negative_samples(all_clicked_items, 
                                                                       additional_samples=negative_sample_recently_clicked_ids, 
                                                                       num_negative_samples=self.negative_samples)
                #Ignoring last elements from second dimension, as they refer to the last labels concatenated with all_clicked_items just to ignore them in negative samples
                batch_negative_items = batch_negative_items[:,:-1,:]                                                                       
                self.batch_negative_items = batch_negative_items

            
            #WARNING: Must keep these variables under the same variable scope, to avoid leaking the positive item to the network (probably due to normalization)
            with tf.variable_scope("user_items_contextual_features"):
                user_context_features = self.get_features(inputs, 
                            features_config=self.session_features_config['sequence_features'],
                            features_to_ignore=SESSION_REQ_SEQ_FEATURES)

                #If there is no user contextual features, creates a dummy variable to not break following concats
                if user_context_features != None:
                    if self.plot_histograms:
                        tf.summary.histogram("user_context_features", user_context_features)
                else:
                    #Dummy tensor with zeroed values
                    user_context_features = tf.zeros_like(tf.expand_dims(item_clicked, -1), dtype=tf.float32)

                
                input_items_features = self.get_item_features(item_clicked, event_timestamp, 'clicked')                                            
                if self.plot_histograms:
                    tf.summary.histogram("input_items_features", input_items_features)

                input_user_items_features_concat = tf.concat([user_context_features, input_items_features], axis=2)
                input_user_items_features = self.scale_center_features(input_user_items_features_concat)
                
                if self.plot_histograms:
                    tf.summary.histogram("input_user_items_features", input_user_items_features)

                input_user_items_features = tf.layers.dropout(input_user_items_features, 
                                                              rate=1.0-self.keep_prob,
                                                              training=self.is_training)
                

                positive_items_features = self.get_item_features(next_item_label, max_event_timestamp, 'positive')
                if self.plot_histograms:
                    tf.summary.histogram("positive_items_features", positive_items_features)
                positive_user_items_features_concat = tf.concat([user_context_features, positive_items_features], axis=2)
                positive_user_items_features = self.scale_center_features(positive_user_items_features_concat)

                if self.plot_histograms:
                    tf.summary.histogram("positive_user_items_features", input_user_items_features)

                positive_user_items_features = tf.layers.dropout(positive_user_items_features, 
                                                              rate=1.0-self.keep_prob,
                                                              training=self.is_training)

                negative_items_features = self.get_item_features(batch_negative_items, max_event_timestamp, 'negative')
                if self.plot_histograms:
                    tf.summary.histogram("negative_items_features", negative_items_features)          
                
                user_context_features_tiled = tf.tile(tf.expand_dims(user_context_features, 2), (1,1,tf.shape(negative_items_features)[2],1))
                negative_user_items_features_concat  = tf.concat([user_context_features_tiled, negative_items_features], axis=3)


                negative_user_items_features = self.scale_center_features(negative_user_items_features_concat, begin_norm_axis=3)
                if self.plot_histograms:
                    tf.summary.histogram("negative_user_items_features", negative_user_items_features)

                negative_user_items_features = tf.layers.dropout(negative_user_items_features, 
                                                              rate=1.0-self.keep_prob,
                                                              training=self.is_training)   


             
            with tf.variable_scope("CAR"):
                PreCAR_dense = tf.layers.Dense(CAR_embedding_size,
                                            activation=tf.nn.leaky_relu, 
                                            kernel_initializer=variance_scaling_initializer(),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay),
                                            name="PreCAR_representation"
                                           )  

                input_contextual_item_embedding_pre_CAR = PreCAR_dense(input_user_items_features)
                
                CAR_dense = tf.layers.Dense(CAR_embedding_size,
                                            activation=tf.nn.tanh, 
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay),
                                            name="CAR_representation"
                                       )
            
            with tf.variable_scope("user_personalized_contextual_article_embedding"):
                with tf.variable_scope("input"):
                    input_contextual_item_embedding = CAR_dense(input_contextual_item_embedding_pre_CAR)
                    
                    if self.plot_histograms:
                        tf.summary.histogram("input_contextual_item_embedding", input_contextual_item_embedding)
                    
                with tf.variable_scope("positive"): 
                    positive_contextual_item_embedding = CAR_dense(PreCAR_dense(positive_user_items_features))
                    if self.plot_histograms:
                        tf.summary.histogram("positive_contextual_item_embedding", positive_contextual_item_embedding)

                with tf.variable_scope("negative"):                                             
                    negative_contextual_item_embedding = CAR_dense(PreCAR_dense(negative_user_items_features))              
                    if self.plot_histograms:
                        tf.summary.histogram("negative_contextual_item_embedding", negative_contextual_item_embedding)                            

            #Building RNN
            rnn_outputs = self.build_rnn(input_contextual_item_embedding, seq_lengths, rnn_units=rnn_units)

            with tf.variable_scope("session_representation"): 
                rnn_outputs_fc1 = tf.layers.dense(rnn_outputs, 512,                                        
                                            activation=tf.nn.leaky_relu, 
                                            kernel_initializer=variance_scaling_initializer(),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay),
                                            name="FC1"
                                           )

                rnn_outputs_fc1_dropout = tf.layers.dropout(inputs=rnn_outputs_fc1, 
                                           rate=1.0-self.keep_prob, 
                                           training=self.is_training)
                
                
                rnn_outputs_fc2 = tf.layers.dense(rnn_outputs_fc1_dropout, CAR_embedding_size,
                    activation=tf.nn.tanh, 
                    name='FC2', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay))

                #tf.summary.scalar('rnn_outputs_fc2/fraction_of_zero_values', tf.nn.zero_fraction(rnn_outputs_fc1))
                
                if self.plot_histograms:
                    tf.summary.histogram("rnn_outputs_fc2", rnn_outputs_fc2)
                
            
            with tf.variable_scope("predicted_contextual_item_embedding"): 
                #Continuing with DSSM losss
                #Apply l2-norm to be able to compute cosine similarity by matrix multiplication
                #predicted_contextual_item_embedding = tf.nn.l2_normalize(rnn_outputs_fc2, axis=-1)
                predicted_contextual_item_embedding = rnn_outputs_fc2
                
                if self.plot_histograms:
                    tf.summary.histogram("predicted_contextual_item_embedding", predicted_contextual_item_embedding)


            with tf.variable_scope("recommendations_ranking"): 

                
                matching_dense_layer_1 = tf.layers.Dense(128,
                                            activation=tf.nn.leaky_relu, 
                                            kernel_initializer=variance_scaling_initializer(),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay),
                                            name="matching_dense_layer_1"
                                       )

                matching_dense_layer_2 = tf.layers.Dense(64,
                                            activation=tf.nn.leaky_relu, 
                                            kernel_initializer=variance_scaling_initializer(),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay),
                                            name="matching_dense_layer_2"
                                       )

                matching_dense_layer_3 = tf.layers.Dense(32,
                                            activation=tf.nn.leaky_relu, 
                                            kernel_initializer=variance_scaling_initializer(), 
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay),
                                            name="matching_dense_layer_3"
                                       )

                matching_dense_layer_4 = tf.layers.Dense(1,
                                            activation=None, 
                                            kernel_initializer=tf.initializers.lecun_uniform(),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay),
                                            name="matching_dense_layer_4"
                                       )


                with tf.variable_scope("cos_sim_positive"):

                    positive_multiplied_embeddings = tf.multiply(positive_contextual_item_embedding, 
                                                                 predicted_contextual_item_embedding)

                    if self.plot_histograms:
                        tf.summary.histogram("train/positive_multiplied_embeddings", positive_multiplied_embeddings)


                    cos_sim_positive = matching_dense_layer_4(matching_dense_layer_3(matching_dense_layer_2(matching_dense_layer_1(positive_multiplied_embeddings))))
                    
                    if self.plot_histograms:
                        tf.summary.histogram("train/cos_sim_positive", 
                                values=tf.boolean_mask(cos_sim_positive, tf.cast(tf.sign(next_item_label), tf.bool)))
                   
                    
                with tf.variable_scope("cos_sim_negative"):
                    negative_multiplied_embeddings = tf.multiply(negative_contextual_item_embedding, 
                                                                 tf.expand_dims(predicted_contextual_item_embedding, 2))
                    
                    if self.plot_histograms:
                        tf.summary.histogram("train/negative_multiplied_embeddings", negative_multiplied_embeddings)

                    cos_sim_negative = matching_dense_layer_4(matching_dense_layer_3(matching_dense_layer_2(matching_dense_layer_1(negative_multiplied_embeddings))))
                    cos_sim_negative = tf.squeeze(cos_sim_negative,  axis=-1)
                    
                    if self.plot_histograms:
                        tf.summary.histogram("train/cos_sim_negative", 
                                values=tf.boolean_mask(cos_sim_negative, tf.cast(tf.sign(next_item_label), tf.bool)))

     
                
                with tf.variable_scope("softmax_function"):                    

                    #Concatenating cosine similarities (positive + K sampled negative)
                    cos_sim_concat = tf.concat([cos_sim_positive, cos_sim_negative], axis=2)                    
                    
                    #Computing softmax over cosine similarities
                    cos_sim_concat_scaled = cos_sim_concat / self.softmax_temperature
                    items_prob = tf.nn.softmax(cos_sim_concat_scaled) 

                    neg_items_prob = tf.nn.softmax(cos_sim_negative / self.softmax_temperature)

                if mode == tf.estimator.ModeKeys.EVAL:
                    #Computing evaluation metrics
                    self.define_eval_metrics(next_item_label, batch_negative_items, items_prob)

                
                
                #if mode == tf.estimator.ModeKeys.TRAIN:
                with tf.variable_scope("samples_popularity"):
                    positive_articles_norm_pop = self.get_items_norm_popularity_feature(next_item_label, summary_suffix='positive')

                    negative_articles_articles_norm_pop = self.get_items_norm_popularity_feature(batch_negative_items, summary_suffix='negative')                    
                    negative_articles_articles_norm_pop_squeezed = tf.squeeze(negative_articles_articles_norm_pop, axis=-1)                                      
                    negative_articles_articles_norm_pop_tiled = negative_articles_articles_norm_pop_squeezed

                    candidate_samples_norm_pop = tf.concat([positive_articles_norm_pop, negative_articles_articles_norm_pop_tiled], axis=2)
                    if self.plot_histograms:
                        tf.summary.histogram("candidate_samples_norm_pop", values=tf.boolean_mask(candidate_samples_norm_pop, tf.cast(tf.sign(next_item_label), tf.bool)))

                    negative_samples_norm_pop_scaled =  self.get_items_pop_novelty_feature(negative_articles_articles_norm_pop_tiled)

                    if self.plot_histograms:
                        tf.summary.histogram("negative_samples_norm_pop_scaled", values=tf.boolean_mask(negative_samples_norm_pop_scaled, tf.cast(tf.sign(next_item_label), tf.bool)))

            with tf.variable_scope("loss"):
                #Computing batch loss
                loss_mask = tf.to_float(self.item_clicked_mask)

                #Computing the probability of the positive item (label)
                positive_prob = items_prob[:,:,0]
                negative_probs = items_prob[:,:,1:]

                
                #Summary of first element of the batch sequence (because others might be masked)
                if self.plot_histograms:
                    tf.summary.histogram("positive_prob", positive_prob[:,0])
                    tf.summary.histogram("negative_probs", negative_probs[:,0,:])


                #reg_loss = self.reg_weight_decay * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if not ("noreg" in tf_var.name or "Bias" in tf_var.name))
                reg_loss = tf.losses.get_regularization_loss()
                tf.summary.scalar("reg_loss", family='train', tensor=reg_loss)
   
                
                #XE loss
                xe_loss = tf.multiply(tf.log(positive_prob), loss_mask)

                
                #Averaging the loss by the number of masked items in the batch
                cosine_sim_loss = -tf.reduce_sum(xe_loss) / tf.reduce_sum(loss_mask) 
                tf.summary.scalar("cosine_sim_loss", family='train', tensor=cosine_sim_loss)

                self.total_loss = cosine_sim_loss + reg_loss

                #if mode == tf.estimator.ModeKeys.TRAIN:
                items_prob_masked = tf.multiply(items_prob, tf.expand_dims(loss_mask, -1), name='items_prob_masked_op')


                if novelty_reg_factor > 0.0:
                    with tf.variable_scope("novelty_loss"):                   
                        masked_nov_reg     = self.novelty_reg_factor * tf.reduce_sum(tf.multiply(tf.multiply(neg_items_prob, negative_samples_norm_pop_scaled), tf.expand_dims(loss_mask, -1)), axis=-1)
                        
                        if self.plot_histograms:
                            tf.summary.histogram("masked_nov_reg", values=tf.boolean_mask(masked_nov_reg, tf.cast(tf.sign(next_item_label), tf.bool))) 
                        
                        nov_reg_loss = tf.reduce_sum(masked_nov_reg) / tf.reduce_sum(loss_mask) 
                                      
                        tf.summary.scalar("nov_reg_loss", family='train', tensor=nov_reg_loss) 
                        self.total_loss = self.total_loss - nov_reg_loss

                tf.summary.scalar("total_loss", family='train', tensor=self.total_loss)
                     
            if mode == tf.estimator.ModeKeys.TRAIN:
                with tf.variable_scope('training'):
                    opt = tf.train.AdamOptimizer(self.lr,
                                                 beta1=0.9,
                                                 beta2=0.999,
                                                 epsilon=1e-08)


                    #Necessary to run update ops for batch_norm, streaming metrics
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)                
                    with tf.control_dependencies(update_ops):      
                        # Get the gradient pairs (Tensor, Variable)
                        grads = opt.compute_gradients(self.total_loss)
                        # Update the weights wrt to the gradient
                        self.train = opt.apply_gradients(grads, 
                                                         global_step=tf.train.get_global_step()#self.gs
                                                        )

                        if self.plot_histograms:
                            # Add histograms for trainable variables.
                            for grad, var in grads:
                                if grad is not None:
                                    tf.summary.histogram(var.op.name + '/gradients', grad)

    def get_features(self, inputs, features_config, 
                             features_to_ignore):

        def cat_ohe(feature_name, size, inputs):
            return tf.one_hot(inputs[feature_name], size, name="{}_cat_one_hot".format(feature_name))
                
        def cat_embed(feature_name, size, inputs):
            with tf.variable_scope("{}_cat_embedding".format(feature_name), reuse=tf.AUTO_REUSE):        
                dim =  get_embedding_size(size)
                embeddings = tf.get_variable("{}_embedding".format(feature_name), shape=[size, dim],
                                             regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay))
                lookup = tf.nn.embedding_lookup(embeddings, ids=inputs[feature_name])#, max_norm=1)
                return lookup

        with tf.variable_scope("features"):
            features_list = []
            for feature_name in features_config:
                #Ignores item_clicked and timestamp as user contextual features
                if feature_name in features_to_ignore:
                    continue

                if features_config[feature_name]['type'] == 'categorical':
                    size = features_config[feature_name]['cardinality']
                    if features_config[feature_name]['cardinality'] <= self.max_cardinality_for_ohe:
                        feature_op = cat_ohe(feature_name, size, inputs)
                    else:
                        feature_op = cat_embed(feature_name, size, inputs)
                elif features_config[feature_name]['type'] == 'numerical':
                    feature_op = tf.expand_dims(inputs[feature_name], -1)
                else:
                    raise Exception('Invalid feature type: {}'.format(feature_name))


                if self.plot_histograms:
                    tf.summary.histogram(feature_name, family='stats',
                                    values=feature_op)

                features_list.append(feature_op)

            if len(features_list) > 0:
                features_concat = tf.concat(features_list, axis=-1)
                return features_concat
            else:
                return None




            
    def define_eval_metrics(self, next_item_label, batch_negative_items, items_prob):
        with tf.variable_scope("evaluation_metrics"):
            
            with tf.variable_scope("predicted_items"):

                next_item_label_expanded = tf.expand_dims(next_item_label, -1)
                
                pos_neg_items_concat = tf.concat([next_item_label_expanded, batch_negative_items], 2)

                #Predicting item ids from [positive + k negative samples]
                items_top_prob = tf.nn.top_k(items_prob,  k=tf.shape(items_prob)[2])
                items_top_prob_indexes = items_top_prob.indices
                self.items_top_prob_values = items_top_prob.values

                items_top_prob_indexes_idx = tf.contrib.layers.dense_to_sparse(items_top_prob_indexes, eos_token=-1).indices
                
                items_top_prob_indexes_val = tf.gather_nd(items_top_prob_indexes, items_top_prob_indexes_idx)
                #Takes the first two columns of the index and use sorted indices as the last column
                items_top_prob_reordered_indexes = tf.concat([items_top_prob_indexes_idx[:,:2], 
                                                              tf.expand_dims(tf.cast(items_top_prob_indexes_val, tf.int64), 1)], 1)
                predicted_item_ids = tf.reshape(tf.gather_nd(pos_neg_items_concat, items_top_prob_reordered_indexes), 
                                                tf.shape(pos_neg_items_concat))
                self.predicted_item_ids = predicted_item_ids



            #Computing Recall@N
            self.recall_at_n, self.recall_at_n_update_op = tf.contrib.metrics.sparse_recall_at_top_k(
                                        labels=next_item_label_expanded,
                                        top_k_predictions=predicted_item_ids[:,:,:self.metrics_top_n],                                                                                              
                                        weights=tf.to_float(self.item_clicked_mask), 
                                        name='hitrate_at_n')


            #Computing MRR@N
            self.mrr, self.mrr_update_op = self.define_mrr_metric(predicted_item_ids, next_item_label_expanded, 
                                                                  topk=self.metrics_top_n)
            
            #Computing NDCG@N
            #self.ndcg_at_n_mean, self.ndcg_at_n_mean_update_op = \
            #    self.define_ndcg_metric(predicted_item_ids, next_item_label_expanded, topk=self.metrics_top_n)


    def define_ndcg_metric(self, predicted_item_ids, next_item_label_expanded, topk):
        with tf.variable_scope("ndcg"):
            #Computing NDCG
            predicted_correct = tf.to_int32(tf.equal(predicted_item_ids, next_item_label_expanded))
            ndcg_predicted = tf_ndcg_at_k(predicted_correct, topk)
            
            #Combining masks of padding items and NDCG zeroed values (because the correct value is not in the top n)
            ndcg_mask = tf.to_float(self.item_clicked_mask)
            
            ndcg_mean, ndcg_mean_update_op = tf.metrics.mean(
                                        values=ndcg_predicted,
                                        weights=ndcg_mask, 
                                        name='ndcg_at_n')              

            return ndcg_mean, ndcg_mean_update_op


    def define_mrr_metric(self, predicted_item_ids, next_item_label_expanded, topk):
        with tf.variable_scope("mrr"):
            reciprocal_ranks = tf.div(tf.constant(1.0), tf.cast(tf.constant(1, tf.int64) + \
                                                                tf.where(
                                                                         tf.logical_and(
                                                                                        tf.equal(next_item_label_expanded,
                                                                                                 predicted_item_ids[:,:,:topk]),
                                                                                        tf.expand_dims(self.item_clicked_mask, -1) #Apply mask to sessions with padded items
                                                                                        )
                                                                        )[:,2],
                                                                tf.float32)) 


            batch_valid_labels_count = tf.reduce_sum(tf.to_int32(self.item_clicked_mask))
            batch_labels_not_found_in_topk = batch_valid_labels_count - tf.size(reciprocal_ranks)


            #Completing with items for which the label was not in the preds (because tf.where() do not return indexes in this case), 
            #so that mean is consistent
            reciprocal_ranks = tf.concat([reciprocal_ranks, tf.zeros(batch_labels_not_found_in_topk)], axis=0)

            
            mrr, mrr_update_op = tf.metrics.mean(
                                        values=reciprocal_ranks,
                                        name='mrr_at_n')              

            return mrr, mrr_update_op

    def scale_center_features(self, item_features, begin_norm_axis=2):


        with tf.variable_scope("input_features_center_scale", reuse=tf.AUTO_REUSE):
            gamma = tf.get_variable("gamma_scale", 
                            shape=[item_features.get_shape()[-1]], 
                            initializer=tf.ones_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay))
            beta = tf.get_variable("beta_center", 
                            shape=[item_features.get_shape()[-1]], 
                            initializer=tf.zeros_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay))

            if self.plot_histograms:
                tf.summary.histogram('input_features_gamma_scale', family='stats', values=gamma)
                tf.summary.histogram('input_features_beta_center', family='stats', values=beta)


            item_features_centered_scaled = (item_features * gamma) + beta

        return item_features_centered_scaled


        
    def items_cat_embed(self, item_ids):
        #with tf.device('/cpu:0'):
        with tf.variable_scope("item_cat_embedding", reuse=tf.AUTO_REUSE):        
            size = self.items_vocab_size
            dim =  get_embedding_size(size)
            embeddings = tf.get_variable("items_embedding", shape=[size, dim],
                                         regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay))
            lookup = tf.nn.embedding_lookup(embeddings, ids=item_ids)#, max_norm=1)
            return lookup
    
    def get_item_features(self, item_ids, events_timestamp, summary_suffix):
        with tf.variable_scope("item_features"):
            
            #Obtaining item features for specified items (e.g. clicked, negative samples)
            item_metadata_features_values = {}
            for feature_name in self.articles_features_config:
                if feature_name not in ARTICLE_REQ_FEATURES:
                    item_metadata_features_values[feature_name] = \
                            tf.gather(self.articles_metadata[feature_name], item_ids)

            items_features_list = []

            if len(item_metadata_features_values) > 0:
                #Concatenating item contextual features
                item_metadata_features = self.get_features(item_metadata_features_values, 
                                features_config=self.articles_features_config, 
                                features_to_ignore=ARTICLE_REQ_FEATURES)
                #Adding articles metadata attributes as input for the network
                items_features_list.append(item_metadata_features)

                if self.plot_histograms:
                    tf.summary.histogram('item_metadata_features/'+summary_suffix, family='stats',
                                        values=tf.boolean_mask(item_metadata_features, tf.cast(tf.sign(item_ids), tf.bool)))


            #If enabled, add Article Content Embeddings trained by ACR module
            if self.internal_features_config['article_content_embeddings']:
                items_acr_embeddings_lookup = tf.nn.embedding_lookup(self.content_article_embeddings_matrix, ids=item_ids)

                items_features_list.append(items_acr_embeddings_lookup)

                if self.plot_histograms:
                    tf.summary.histogram('items_acr_embeddings_lookup/'+summary_suffix, family='stats',
                                    values=tf.boolean_mask(items_acr_embeddings_lookup, tf.cast(tf.sign(item_ids), tf.bool)))                 
                

            #If enabled, adds trainable item embeddings
            if self.internal_features_config['item_clicked_embeddings']:
                item_clicked_interactions_embedding = self.items_cat_embed(item_ids)
                items_features_list.append(item_clicked_interactions_embedding)

                if self.plot_histograms:
                    tf.summary.histogram('item_clicked_interactions_embedding/'+summary_suffix, family='stats',
                                    values=tf.boolean_mask(item_clicked_interactions_embedding, tf.cast(tf.sign(item_ids), tf.bool)))

            #Computing Item Dynamic features (RECENCY and POPULARITY)
            items_context_features = self.get_items_dynamic_features(item_ids, 
                                                        events_timestamp, 
                                                        summary_suffix=summary_suffix)
            
            #If both Recency and Novelty feature were disabled, ignore them
            if items_context_features is not None:
                items_features_list.append(items_context_features)

            items_features_concat = tf.concat(items_features_list, axis=-1)
            
            return items_features_concat

    def min_max_normalization(self, tensor, tensor_to_get_stats_from, min_max_range=(-1.0,1.0), epsilon=1e-24, summary_suffix=''):
        epsilon = tf.constant(epsilon, dtype=tf.float32, name="epsilon_min_max")
        min_scale = tf.constant(min_max_range[0], dtype=tf.float32, name="scale_min_value")
        max_scale = tf.constant(min_max_range[1], dtype=tf.float32, name="scale_max_value")

        min_value = tf.reduce_min(tensor_to_get_stats_from)
        max_value = tf.reduce_max(tensor_to_get_stats_from)

        tf.summary.scalar('min_max_normalization/'+summary_suffix+'/min', family='stats', tensor=min_value)
        tf.summary.scalar('min_max_normalization/'+summary_suffix+'/max', family='stats', tensor=max_value)

        scaled = (tensor - min_value + epsilon) / tf.maximum((max_value - min_value), 2*epsilon)
        centered = scaled * (max_scale - min_scale) + min_scale
        return centered
       
    def normalize_values(self, tensor_to_normalize, tensor_to_get_stats_from, summary_suffix='', 
                         min_max_scaling_after_znorm=True, min_max_range=(-1.0,1.0)): 
        with tf.variable_scope("values_normalization"):       
            mean, variance  = tf.nn.moments(tensor_to_get_stats_from, axes=[0])        

            #tf.logging.info('normalize_values/{}/mean={}'.format(summary_suffix, mean.get_shape()))
            #tf.logging.info('normalize_values/{}/variance={}'.format(summary_suffix, variance.get_shape()))
            
            #Fixing size of stats to avoid dynamic last dimension on tensor_normed
            mean = tf.reshape(mean, [1])
            variance = tf.reshape(variance, [1])
             
            #To avoid division by zero
            epsilon = tf.constant(1e-24)
            stddev = tf.sqrt(variance + epsilon)

            tf.summary.scalar('normalize_values/'+summary_suffix+'/mean', family='stats', tensor=mean[0])
            tf.summary.scalar('normalize_values/'+summary_suffix+'/stddev', family='stats', tensor=stddev[0])
                 
            #Standardization (z-normalization)
            tensor_normed = (tensor_to_normalize - mean)  / stddev

            if min_max_scaling_after_znorm:
                tensor_to_get_stats_from_normed = (tensor_to_get_stats_from - mean)  / stddev
                tensor_normed = self.min_max_normalization(tensor_normed, tensor_to_get_stats_from_normed, 
                                                            min_max_range=min_max_range,
                                                            summary_suffix=summary_suffix)

            return tensor_normed    
    
    def get_last_items_from_recent_clicks_buffer(self, last_n):
        with tf.variable_scope("last_items_from_recent_clicks_buffer"):   
            non_zero_recent_items = tf.boolean_mask(self.pop_recent_items_buffer, tf.cast(tf.sign(self.pop_recent_items_buffer), tf.bool))
            return non_zero_recent_items[:last_n]

    def get_unique_items_from_pop_recent_buffer(self):
        with tf.variable_scope("unique_items_from_pop_recent_buffer"):    
            recent_items_unique, _ = tf.unique(self.pop_recent_items_buffer)
            #Removing zero
            recent_items_unique = tf.boolean_mask(recent_items_unique,
                                                  tf.cast(tf.sign(recent_items_unique), tf.bool)) 
            return recent_items_unique

    
    def calculate_elapsed_days_since_publishing(self, creation_dates, reference_timestamps):
        with tf.variable_scope("elapsed_days_since_publishing"):
            #Timestamps and created_at_ts
            elapsed_days = tf.nn.relu((tf.to_float(reference_timestamps) \
                                      - tf.to_float(creation_dates)) / tf.constant(1000.0 * 60.0 * 60.0 * 24.0))
            return elapsed_days
    
    def normalize_recency_feature(self, batch_elapsed_days_since_publishing, batch_events_timestamp, item_ids, 
                                        summary_suffix=''):
        with tf.variable_scope("normalize_recency_feature"):   
            #Computing global recency stats from buffer
            last_clicked_items = self.get_last_items_from_recent_clicks_buffer(self.recent_clicks_for_normalization)
            recent_items_creation_date = tf.gather(self.articles_metadata['created_at_ts'], last_clicked_items)
            recent_items_elapsed_days_since_creation= self.calculate_elapsed_days_since_publishing(recent_items_creation_date, 
                                                                                    tf.reduce_max(batch_events_timestamp))
            recent_items_elapsed_days_since_creation_smoothed = log_1p(recent_items_elapsed_days_since_creation, 
                                                                       base=self.elapsed_days_smooth_log_base)                                                                           

            #Normalizing batch recency feature
            batch_elapsed_days_since_publishing_smoothed = log_1p(batch_elapsed_days_since_publishing, 
                                                                  base=self.elapsed_days_smooth_log_base)


            batch_elapsed_days_since_publishing_smoothed_non_zero = tf.reshape(tf.boolean_mask(batch_elapsed_days_since_publishing_smoothed, tf.cast(tf.sign(item_ids), tf.bool)), [-1])
            
            #If there aren't recent items available in the buffer (first batch), use batch items (zeroed matrix) to compute norm stats
            #After that, do not use batch to compute mean and stddev, to avoid leak
            tensor_to_get_stats_from = tf.cond(tf.equal(tf.shape(last_clicked_items)[0], tf.constant(0)), 
                                               lambda: batch_elapsed_days_since_publishing_smoothed_non_zero,
                                               lambda: recent_items_elapsed_days_since_creation_smoothed)
                       
            batch_elapsed_days_since_publishing_normed = self.normalize_values(batch_elapsed_days_since_publishing_smoothed, 
                                                                               tensor_to_get_stats_from,
                                                                               summary_suffix='log_elapsed_days_since_publishing/'+summary_suffix)
            return batch_elapsed_days_since_publishing_normed
        
    
    def get_items_recency_feature(self, item_ids, events_timestamp, summary_suffix=''):
        with tf.variable_scope("items_recency_feature"):  
            #Computing RECENCY feature
            batch_articles_creation_date = tf.gather(tf.reshape(self.articles_metadata['created_at_ts'], 
                                                                [-1,1]), item_ids)

            '''
            #TO DEBUG
            elapsed_hours = self.calculate_elapsed_hours_temp(batch_articles_creation_date, events_timestamp)
            if self.plot_histograms:
                tf.summary.histogram('batch_elapsed_hours/'+summary_suffix, family='stats',
                                  values=tf.boolean_mask(elapsed_hours, tf.cast(tf.sign(item_ids), tf.bool)))

                tf.summary.scalar('batch_elapsed_hours_scalar/'+summary_suffix, family='stats',
                                    tensor=tf.reduce_mean(tf.boolean_mask(elapsed_hours, tf.cast(tf.sign(item_ids), tf.bool))))
            '''
                                                            
            elapsed_days_since_publishing = self.calculate_elapsed_days_since_publishing(batch_articles_creation_date, events_timestamp)
            
            tf.summary.scalar('batch_elapsed_days_since_publishing_scalar/'+summary_suffix, family='stats',
                                    tensor=tf.reduce_mean(tf.boolean_mask(elapsed_days_since_publishing, tf.cast(tf.sign(item_ids), tf.bool))))
                                    
            if self.plot_histograms:
                tf.summary.histogram('batch_elapsed_days_since_publishing/'+summary_suffix, family='stats',
                                  values=tf.boolean_mask(elapsed_days_since_publishing, tf.cast(tf.sign(item_ids), tf.bool)))
            
            
            elapsed_days_since_publishing_norm = self.normalize_recency_feature(elapsed_days_since_publishing, 
                                                                                  events_timestamp, item_ids, 
                                                                                  summary_suffix=summary_suffix)

            tf.summary.scalar('batch_elapsed_days_since_publishing_norm_scalar/'+summary_suffix, family='stats',
                                    tensor=tf.reduce_mean(tf.boolean_mask(elapsed_days_since_publishing_norm, tf.cast(tf.sign(item_ids), tf.bool))))
            
            if self.plot_histograms:
                tf.summary.histogram('batch_elapsed_days_since_publishing_norm/'+summary_suffix, family='stats',
                                  values=tf.boolean_mask(elapsed_days_since_publishing_norm, tf.cast(tf.sign(item_ids), tf.bool)))                
                    
            
            return elapsed_days_since_publishing_norm, batch_articles_creation_date


    def get_items_norm_popularity_feature(self, item_ids, summary_suffix=''):
        #Computing POPULARITY feature
        with tf.variable_scope("items_norm_popularity_feature"):         

            batch_articles_norm_pop = tf.gather(self.articles_recent_pop_norm, tf.expand_dims(item_ids, -1))

            if self.plot_histograms:
                tf.summary.histogram('batch_articles_norm_pop/'+summary_suffix, family='stats',
                                  values=tf.boolean_mask(batch_articles_norm_pop, tf.cast(tf.sign(item_ids), tf.bool)))

            
            return batch_articles_norm_pop

    def get_items_pop_novelty_feature(self, items_norm_pop):
        return -log_base(items_norm_pop, self.popularity_smooth_log_base)

    def get_items_pop_novelty_feature_standardized(self, item_ids, summary_suffix=''):
        #Computing POPULARITY feature
        with tf.variable_scope("items_novelty_feature"):   

            #Computing global recency stats from buffer
            #recent_items_unique = self.get_unique_items_from_pop_recent_buffer()            
            last_clicked_items = self.get_last_items_from_recent_clicks_buffer(self.recent_clicks_for_normalization)
            recent_items_norm_pop = self.get_items_norm_popularity_feature(last_clicked_items, summary_suffix=summary_suffix+'_global')
            recent_items_novelty = self.get_items_pop_novelty_feature(recent_items_norm_pop)

            tf.summary.scalar('recent_items_novelty/'+summary_suffix, family='stats',
                                    tensor=tf.reduce_mean(recent_items_novelty))


            batch_articles_norm_pop_input = self.get_items_norm_popularity_feature(item_ids, summary_suffix=summary_suffix)

            batch_articles_novelty = self.get_items_pop_novelty_feature(batch_articles_norm_pop_input)

            batch_articles_novelty_non_zero = tf.boolean_mask(batch_articles_novelty, tf.cast(tf.sign(item_ids), tf.bool))
                        
            if self.plot_histograms:
                tf.summary.histogram('batch_articles_novelty/'+summary_suffix, family='stats',
                                  values=batch_articles_novelty_non_zero)

            tf.summary.scalar('batch_items_novelty/'+summary_suffix, family='stats',
                                    tensor=tf.reduce_mean(batch_articles_novelty_non_zero))

            #If there aren't recent items available in the buffer (first batch), use batch items (zeroed matrix) to compute norm stats
            #After that, do not use batch to compute mean and stddev, to avoid leak
            tensor_to_get_stats_from = tf.cond(tf.equal(tf.shape(last_clicked_items)[0], tf.constant(0)), 
                                               lambda: batch_articles_novelty_non_zero,
                                               lambda: recent_items_novelty)
            
            #Applying standardization
            batch_items_novelty_standardized = self.normalize_values(batch_articles_novelty, 
                                                        tensor_to_get_stats_from, 
                                                        summary_suffix='novelty/'+summary_suffix)

            if self.plot_histograms:
                tf.summary.histogram('batch_items_novelty_standardized/'+summary_suffix, family='stats',
                                  values=batch_items_novelty_standardized)

            
            return batch_items_novelty_standardized

                                 
    
    def get_items_dynamic_features(self, item_ids, events_timestamp, summary_suffix=''):
        
        with tf.variable_scope("items_dynamic_features", reuse=tf.AUTO_REUSE):
        
            #Computing RECENCY feature
            elapsed_days_since_publishing_log, batch_articles_creation_date = \
                        self.get_items_recency_feature(item_ids, events_timestamp, summary_suffix=summary_suffix)

            batch_articles_novelty = self.get_items_pop_novelty_feature_standardized(item_ids,
                                                                       summary_suffix=summary_suffix) 

            #Including dynamic item features, according to configuration
            dynamic_features = []
            if self.internal_features_config['recency']:
                dynamic_features.append(elapsed_days_since_publishing_log)
            if self.internal_features_config['novelty']:
                dynamic_features.append(batch_articles_novelty)

            if len(dynamic_features) > 0:
                return tf.concat(dynamic_features, axis=-1)
            else:
                return None
            
    def get_sample_from_recently_clicked_items_buffer(self, sample_size):
        with tf.variable_scope("neg_samples_buffer"):
            pop_recent_items_buffer_masked = tf.boolean_mask(self.pop_recent_items_buffer,
                                                      tf.cast(tf.sign(self.pop_recent_items_buffer), tf.bool)) 
            
            #tf.summary.scalar('unique_clicked_items_on_buffer', family='stats', tensor=tf.shape(unique_pop_recent_items_buffer_masked)[0])
            tf.summary.scalar('clicked_items_on_buffer', family='stats', tensor=tf.shape(pop_recent_items_buffer_masked)[0])
            
            #recent_items_unique_sample, idxs = tf.unique(tf.random_shuffle(pop_recent_items_buffer_masked)[:sample_size*sample_size_factor_to_look_for_unique])
            recent_items_unique_sample = tf.random_shuffle(pop_recent_items_buffer_masked)
            
            #Samples K articles from recent articles
            sample_recently_clicked_items = recent_items_unique_sample[:sample_size]
            return sample_recently_clicked_items
    

    def get_masked_seq_values(self, tensor):
        return tf.boolean_mask(tensor, self.item_clicked_mask, name='masked_values')      

    def get_neg_items_click(self, valid_samples_session, num_neg_samples):
        #Shuffles neg. samples for each click
        valid_samples_shuffled = tf.random.shuffle(valid_samples_session)

        
        samples_unique_vals, samples_unique_idx = tf.unique(valid_samples_shuffled)

        #Returning first N unique items (to avoid repetition)
        first_unique_items = tf.unsorted_segment_min(data=valid_samples_shuffled,                   
                                           segment_ids=samples_unique_idx,                        
                                           num_segments=tf.shape(samples_unique_vals)[0])[:num_neg_samples]

        #Padding if necessary to keep the number of neg samples constant (ex: first batch)
        first_unique_items_padded_if_needed = tf.concat([first_unique_items, tf.zeros(num_neg_samples-tf.shape(first_unique_items)[0], tf.int64)], axis=0)

        return first_unique_items_padded_if_needed                            


    def get_neg_items_session(self, session_item_ids, candidate_samples, num_neg_samples):
        #Ignoring negative samples clicked within the session (keeps the order and repetition of candidate_samples)
        valid_samples_session, _ = tf.setdiff1d(candidate_samples, session_item_ids, index_dtype=tf.int64)

        #Generating a random list of negative samples for each click (with no repetition)
        session_clicks_neg_items = tf.map_fn(lambda click_id: tf.cond(tf.equal(click_id, tf.constant(0, tf.int64)), 
                                                                      lambda: tf.zeros(num_neg_samples, tf.int64),
                                                                      lambda: self.get_neg_items_click(valid_samples_session, num_neg_samples)
                                                                      )
                                             , session_item_ids)                                                     

        return session_clicks_neg_items
        

    def get_negative_samples(self, all_clicked_items, candidate_samples, num_neg_samples):  
        with tf.variable_scope("negative_samples"):      
            #Shuffling negative samples by session and limiting to num_neg_samples
            shuffled_neg_samples = tf.map_fn(lambda session_item_ids: self.get_neg_items_session(session_item_ids, 
                                                                                                 candidate_samples, 
                                                                                                 num_neg_samples)
                                            , all_clicked_items)

            return shuffled_neg_samples
         
    def get_batch_negative_samples(self, all_clicked_items, additional_samples, num_negative_samples, 
                                         first_sampling_multiplying_factor=20):
        with tf.variable_scope("neg_samples_batch"):
            #current_batch_size, batch_max_session_length = tf.shape(item_clicked)[0], tf.shape(item_clicked)[1] 

            batch_items = tf.reshape(all_clicked_items, [-1])
            
            #Removing padded (zeroed) items
            batch_items_non_zero = tf.boolean_mask(batch_items, tf.cast(tf.sign(batch_items), dtype=tf.bool))

            #TEMP: uniform sampling -> popularity sampling
            #batch_items_unique, _ = tf.unique(batch_items_non_zero)
            batch_items_unique = batch_items_non_zero
            
                            
            #Concatenating batch items with additional samples (to deal with small batches)
            candidate_neg_items = tf.concat([batch_items_unique, additional_samples], axis=0)     

            #Shuffling candidates and sampling the first 20N (1000 if neg_samples=50)
            candidate_neg_items_shuffled = tf.random.shuffle(candidate_neg_items)[:(num_negative_samples*first_sampling_multiplying_factor)]

            batch_negative_items = self.get_negative_samples(all_clicked_items, candidate_neg_items_shuffled, num_negative_samples) 
            
            return batch_negative_items

    
    #Good reference: https://github.com/tensorflow/magenta/blob/master/magenta/models/shared/events_rnn_graph.py
    def build_rnn(self, the_input, lengths, rnn_units=256, residual_connections=False):    
        with tf.variable_scope("RNN"):    
            fw_cells = []
            #bw_cells = []

            #Hint: Use tf.contrib.rnn.InputProjectionWrapper if the number of units between layers is different
            for i in range(self.rnn_num_layers):
                #cell = tf.nn.rnn_cell.GRUCell(rnn_units)  
                #cell = tf.nn.rnn_cell.LSTMCell(rnn_units, state_is_tuple=True)        
                cell = tf.contrib.rnn.UGRNNCell(rnn_units) 

                if residual_connections:
                    cell = tf.contrib.rnn.ResidualWrapper(cell)
                    if i == 0: #or rnn_layer_sizes[i] != rnn_layer_sizes[i - 1]:
                        #cell = tf.contrib.rnn.InputProjectionWrapper(cell, rnn_layer_sizes[i])  
                        cell = tf.contrib.rnn.InputProjectionWrapper(cell, rnn_units)  
                
                '''
                cell = tf.contrib.rnn.AttentionCellWrapper(cell, 
                                                     attn_length=20,
                                                     state_is_tuple=True)
                '''

                cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                                     output_keep_prob=self.keep_prob 
                                                     #, input_keep_prob=self.keep_prob
                                                     )
                fw_cells.append(cell)   


            fw_stacked_cells = tf.contrib.rnn.MultiRNNCell(fw_cells, state_is_tuple=True)
            #bw_stacked_cells = tf.contrib.rnn.MultiRNNCell(bw_cells, state_is_tuple=True)

            rnn_outputs, rnn_final_hidden_state_tuples = \
                tf.nn.dynamic_rnn(fw_stacked_cells, the_input, dtype=tf.float32, sequence_length=lengths)

            '''            
            outputs, states  = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_stacked_cells,
                cell_bw=bw_stacked_cells,
                dtype=tf.float32, #tf.float64,
                sequence_length=lengths,
                inputs=the_input)

            #output_fw, output_bw = outputs
            #states_fw, states_bw = states
            #last_lstm_output = output_fw[:,-1,:]
            rnn_outputs = tf.concat(outputs, axis=2)
            '''

            if self.plot_histograms:
                tf.summary.histogram("rnn/outputs", rnn_outputs)   
            
            return rnn_outputs
    
    #def create_item_embed_lookup_variable(self):        
    #    with tf.variable_scope("item_embedding"):          
    #        self.content_embedding_variable = tf.Variable(self.content_article_embeddings_matrix, #collections=[],
    #                                                      trainable=False)








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
        #Stores the timestamp of the first click in the item
        self.items_first_click_ts = dict()
        #Stores the delay (in minutes) from item's first click to item's first recommendation from CHAMELEON
        self.items_delay_for_first_recommendation = dict()
        #States specific for benchmarks
        self.benchmarks_states = dict()

        
    def save_state_checkpoint(self):
        self.articles_pop_chkp = np.copy(self.articles_pop)
        self.pop_recent_clicks_buffer_chkp = np.copy(self.pop_recent_clicks_buffer)
        self.items_coocurrences_chkp = csr_matrix.copy(self.items_coocurrences)
        #self.items_coocurrences_chkp = lil_matrix.copy(self.items_coocurrences)
        self.items_first_click_ts_chkp = deepcopy(self.items_first_click_ts)
        self.items_delay_for_first_recommendation_chkp = deepcopy(self.items_delay_for_first_recommendation)
        self.benchmarks_states_chkp = deepcopy(self.benchmarks_states)
        
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
    
    def update_items_state(self, batch_clicked_items, batch_clicked_timestamps):
        #batch_items_nonzero = self._get_non_zero_items_vector(batch_clicked_items)

        self._update_recently_clicked_items_buffer(batch_clicked_items, batch_clicked_timestamps)
        self._update_recent_pop_items()

        self._update_pop_items(batch_clicked_items)        
            
    
    def _update_recently_clicked_items_buffer(self, batch_clicked_items, batch_clicked_timestamps):
        MILISECS_BY_HOUR = 1000 * 60 * 60

        #Concatenating column vectors of batch clicked items
        batch_recent_clicks_timestamps = np.hstack([batch_clicked_items.reshape(-1,1), batch_clicked_timestamps.reshape(-1,1)])
        #Inverting the order of clicks, so that latter clicks are now the first in the vector
        batch_recent_clicks_timestamps = batch_recent_clicks_timestamps[::-1]
        
        #Keeping in the buffer only clicks within the last N hours
        min_timestamp_batch = np.min(batch_clicked_timestamps)
        min_timestamp_buffer_threshold = min_timestamp_batch - int(self.recent_clicks_buffer_hours * MILISECS_BY_HOUR)
        self.pop_recent_clicks_buffer = self.pop_recent_clicks_buffer[self.pop_recent_clicks_buffer[:,self.pop_recent_buffer_timestamp_column]>=min_timestamp_buffer_threshold]
        #Concatenating batch clicks with recent buffer clicks, limited by the buffer size
        self.pop_recent_clicks_buffer = np.vstack([batch_recent_clicks_timestamps, self.pop_recent_clicks_buffer])[:self.recent_clicks_buffer_max_size]
        #Complete buffer with zeroes if necessary
        if self.pop_recent_clicks_buffer.shape[0] < self.recent_clicks_buffer_max_size:
            self.pop_recent_clicks_buffer = np.vstack([self.pop_recent_clicks_buffer, 
                                                       np.zeros(shape=[self.recent_clicks_buffer_max_size-self.pop_recent_clicks_buffer.shape[0], 2], dtype=np.int64)])
        

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





class ItemsStateUpdaterHook(tf.train.SessionRunHook):
    """Saves summaries during eval loop."""

    def __init__(self, mode, model, eval_metrics_top_n, 
                 clicked_items_state, eval_sessions_metrics_log,
                 sessions_negative_items_log,
                 sessions_chameleon_recommendations_log,
                 content_article_embeddings_matrix,
                 articles_metadata,
                 eval_negative_sample_relevance,
                 eval_benchmark_classifiers=[],
                 eval_metrics_by_session_position=False):

        self.mode = mode
        self.model = model        
        self.eval_metrics_top_n = eval_metrics_top_n
                
        self.clicked_items_state = clicked_items_state
        self.eval_sessions_metrics_log = eval_sessions_metrics_log
        self.sessions_negative_items_log = sessions_negative_items_log
        self.sessions_chameleon_recommendations_log = sessions_chameleon_recommendations_log

        self.content_article_embeddings_matrix = content_article_embeddings_matrix
        self.articles_metadata = articles_metadata
        self.eval_negative_sample_relevance = eval_negative_sample_relevance
        self.eval_metrics_by_session_position = eval_metrics_by_session_position

        self.bench_classifiers = list([clf['recommender'](self.clicked_items_state,
                                                     clf['params'],
                                                     ItemsStateUpdaterHook.create_eval_metrics(self.eval_metrics_top_n, 
                                                                                               self.eval_negative_sample_relevance,
                                                                                               #False, 
                                                                                               self.eval_metrics_by_session_position,
                                                                                               self.content_article_embeddings_matrix,
                                                                                               self.articles_metadata,
                                                                                               self.clicked_items_state)) for clf in eval_benchmark_classifiers])
        

    def begin(self):        
        if self.mode == tf.estimator.ModeKeys.EVAL:

            tf.logging.info("Saving items state checkpoint from train")
            #Save state of items popularity and recency from train loop, to restore after evaluation finishes
            self.clicked_items_state.save_state_checkpoint()  
            
            #Resets streaming metrics
            self.eval_streaming_metrics_last = {}            
            for clf in self.bench_classifiers:
                clf.reset_eval_metrics()

            self.streaming_metrics = ItemsStateUpdaterHook.create_eval_metrics(self.eval_metrics_top_n, 
                                                                               self.eval_negative_sample_relevance,
                                                                               self.eval_metrics_by_session_position,
                                                                               self.content_article_embeddings_matrix,
                                                                               self.articles_metadata,
                                                                               self.clicked_items_state)
            #self.metrics_by_session_pos = StreamingMetrics(topn=self.metrics_top_n)
                
            self.stats_logs = []


    #Runs before every batch
    def before_run(self, run_context): 
        fetches = {'clicked_items': self.model.item_clicked,
                   'clicked_timestamps': self.model.event_timestamp,
                   'next_item_labels': self.model.next_item_label,
                   'last_item_label': self.model.label_last_item,                   
                   'session_id': self.model.session_id,
                   #'session_start': self.model.session_start,
                   'user_id': self.model.user_id,
                   }

        #Commenting to improve performance during training
        #fetches['predicted_item_ids'] = self.model.predicted_item_ids

        if self.mode == tf.estimator.ModeKeys.EVAL:
            fetches['eval_batch_negative_items'] = self.model.batch_negative_items
            fetches['batch_items_count'] = self.model.batch_items_count
            fetches['batch_unique_items_count'] = self.model.batch_unique_items_count

            fetches['hitrate_at_n'] = self.model.recall_at_n_update_op
            fetches['mrr_at_n'] = self.model.mrr_update_op
            #fetches['ndcg_at_n'] = self.model.ndcg_at_n_mean_update_op
            
            fetches['predicted_item_ids'] = self.model.predicted_item_ids
            fetches['predicted_item_probs'] = self.model.items_top_prob_values
        
        feed_dict = {
            self.model.articles_recent_pop_norm: self.clicked_items_state.get_articles_recent_pop_norm(),            
            self.model.pop_recent_items_buffer:  self.clicked_items_state.get_recent_clicks_buffer(), 
            #Passed as placeholder (and not as a constant) to avoid been saved in checkpoints
            self.model.content_article_embeddings_matrix: self.content_article_embeddings_matrix           
        }         

        #Passed as placeholder (and not as a constant) to avoid been saved in checkpoints
        for feature_name in self.articles_metadata:
            feed_dict[self.model.articles_metadata[feature_name]] = self.articles_metadata[feature_name]

        return tf.train.SessionRunArgs(fetches=fetches,
                                       feed_dict=feed_dict)
    
    
    def evaluate_and_update_streaming_metrics_last(self, clf, users_ids, clicked_items, next_item_labels, eval_negative_items):
        tf.logging.info('Evaluating benchmark: {}'.format(clf.get_description()))  
        clf_metrics = clf.evaluate(users_ids, clicked_items, next_item_labels, topk=self.eval_metrics_top_n, 
                                   eval_negative_items=eval_negative_items)
        self.eval_streaming_metrics_last = merge_two_dicts(self.eval_streaming_metrics_last, clf_metrics)

    #Runs after every batch
    def after_run(self, run_context, run_values):     
        clicked_items = run_values.results['clicked_items']
        clicked_timestamps = np.squeeze(run_values.results['clicked_timestamps'], axis=-1)
        next_item_labels = run_values.results['next_item_labels']
        last_item_label = run_values.results['last_item_label'] 

        users_ids = run_values.results['user_id']
        sessions_ids = run_values.results['session_id']

                
        if self.mode == tf.estimator.ModeKeys.EVAL:
            self.eval_streaming_metrics_last = {}
            #self.eval_streaming_metrics_last['hitrate_at_1'] = run_values.results['hitrate_at_1']
            self.eval_streaming_metrics_last['hitrate_at_n'] = run_values.results['hitrate_at_n']
            self.eval_streaming_metrics_last['mrr_at_n'] = run_values.results['mrr_at_n']
            #self.eval_streaming_metrics_last['ndcg_at_n'] = run_values.results['ndcg_at_n']

            #Only for eval, to improve performance during training 
            predicted_item_ids = run_values.results['predicted_item_ids']
            tf.logging.info('predicted_item_ids (shape): {}'.format(predicted_item_ids.shape))  
           
            
            eval_batch_negative_items = run_values.results['eval_batch_negative_items']
            predicted_item_probs = run_values.results['predicted_item_probs']

            if self.sessions_negative_items_log != None:
                #Acumulating session negative items, to allow evaluation comparison
                # with benchmarks outsite the framework (e.g. Matrix Factorization) 
                for session_id, labels, neg_items in zip(sessions_ids,
                                                 next_item_labels,
                                                 eval_batch_negative_items):  

                    self.sessions_negative_items_log.append({'session_id': str(session_id), #Convert numeric session_id to str because large ints are not serializable
                                                         'negative_items': list([neg_items_click for label, neg_items_click in zip(labels.tolist(), 
                                                                                                                                   neg_items.tolist()) if label != 0])})


            if self.sessions_chameleon_recommendations_log != None:
                predicted_item_probs = run_values.results['predicted_item_probs']
                predicted_item_probs_rounded = predicted_item_probs.round(decimals=7)

                articles_recent_pop_norm = self.clicked_items_state.get_articles_recent_pop_norm()

                #Acumulating CHAMELEON predictions, labels, scores, accuracy, novelty and diversity to allow greed re-ranking approachs (e.g. MMR)
                for session_id, labels, pred_item_ids, pred_item_probs \
                                           in zip(sessions_ids,
                                                 next_item_labels, 
                                                 predicted_item_ids,
                                                 predicted_item_probs_rounded
                                                 ):   

                    #Reducing the precision to 5 decimals for serialization
                    pred_item_norm_pops = articles_recent_pop_norm[pred_item_ids].round(decimals=7)

                    labels_filtered = []
                    pred_item_ids_filtered = []
                    pred_item_probs_filtered = []
                    pred_item_norm_pops_filtered = []

                    for label, pred_item_ids_click, pred_item_probs_click, pred_item_norm_pops_click \
                                                        in zip(labels.tolist(),
                                                               pred_item_ids.tolist(),
                                                               pred_item_probs.tolist(),
                                                               pred_item_norm_pops.tolist()):
                        if label != 0:
                            labels_filtered.append(label)
                            pred_item_ids_filtered.append(pred_item_ids_click)
                            pred_item_probs_filtered.append(pred_item_probs_click)
                            pred_item_norm_pops_filtered.append(pred_item_norm_pops_click)


                    to_append = {'session_id': str(session_id), #Convert numeric session_id to str because large ints are not serializable
                                                                'next_click_labels': labels_filtered,
                                                                'predicted_item_ids': pred_item_ids_filtered,
                                                                'predicted_item_probs': pred_item_probs_filtered,
                                                                'predicted_item_norm_pop': pred_item_norm_pops_filtered
                                                                }
                    self.sessions_chameleon_recommendations_log.append(to_append) 

            batch_stats = {#'eval_sampled_negative_items': eval_batch_negative_items.shape[1],
                           'batch_items_count': run_values.results['batch_items_count'],
                           'batch_unique_items_count': run_values.results['batch_unique_items_count'],
                           'batch_sessions_count': len(sessions_ids)
                          }
            self.stats_logs.append(batch_stats)
            tf.logging.info('batch_stats: {}'.format(batch_stats))

            preds_norm_pop = self.clicked_items_state.get_articles_recent_pop_norm()[predicted_item_ids]

            
            labels_norm_pop = self.clicked_items_state.get_articles_recent_pop_norm()[next_item_labels]

            #Computing metrics for this neural model
            update_metrics(predicted_item_ids, next_item_labels, labels_norm_pop, preds_norm_pop, clicked_items,
                                            self.streaming_metrics, 
                                            recommender='chameleon')  
            model_metrics_values = compute_metrics_results(self.streaming_metrics, 
                                            recommender='chameleon')            
            self.eval_streaming_metrics_last = merge_two_dicts(self.eval_streaming_metrics_last, 
                                                               model_metrics_values)
            
            
            
            start_eval = time()
            #Computing metrics for Benchmark recommenders
            for clf in self.bench_classifiers:                    
                self.evaluate_and_update_streaming_metrics_last(clf, users_ids, 
                                clicked_items, next_item_labels, eval_batch_negative_items)
            tf.logging.info('Total elapsed time evaluating benchmarks: {}'.format(time() - start_eval))            

            tf.logging.info('Finished benchmarks evaluation')

        #Training benchmark classifier
        for clf in self.bench_classifiers:
            #It is required that session_ids are sorted by time (ex: first_timestamp+hash_session_id), so that
            #recommenders that trust in session_id to sort by recency work (e.g. V-SkNN)
            clf.train(users_ids, sessions_ids, clicked_items, next_item_labels)
        

        #Concatenating all clicked items in the batch (including last label)
        batch_clicked_items = np.concatenate([clicked_items,last_item_label], axis=1)
        #Flattening values and removing padding items (zeroes) 
        batch_clicked_items_flatten = batch_clicked_items.reshape(-1)
        batch_clicked_items_nonzero = batch_clicked_items_flatten[np.nonzero(batch_clicked_items_flatten)]
        
        #As timestamp of last clicks are not available for each session, assuming they are the same than previous session click
        last_timestamp_batch = np.max(clicked_timestamps, axis=1).reshape(-1,1)
        batch_clicked_timestamps = np.concatenate([clicked_timestamps,last_timestamp_batch], axis=1)
        #Flattening values and removing padding items (zeroes)        
        batch_clicked_timestamps_flatten = batch_clicked_timestamps.reshape(-1)        
        batch_clicked_timestamps_nonzero = batch_clicked_timestamps_flatten[np.nonzero(batch_clicked_items_flatten)]

        #Updating items state
        self.clicked_items_state.update_items_state(batch_clicked_items_nonzero, batch_clicked_timestamps_nonzero)        
        self.clicked_items_state.update_items_coocurrences(batch_clicked_items)
 
    
    def end(self, session=None):
        if self.mode == tf.estimator.ModeKeys.EVAL:    
            #avg_neg_items = np.mean([x['eval_sampled_negative_items'] for x in self.stats_logs])
            #self.eval_streaming_metrics_last['avg_eval_sampled_neg_items'] = avg_neg_items
            
            clicks_count = np.sum([x['batch_items_count'] for x in self.stats_logs])
            self.eval_streaming_metrics_last['clicks_count'] = clicks_count

            sessions_count = np.sum([x['batch_sessions_count'] for x in self.stats_logs])
            self.eval_streaming_metrics_last['sessions_count'] = sessions_count
                        
            self.eval_sessions_metrics_log.append(self.eval_streaming_metrics_last)
            eval_metrics_str = '\n'.join(["'{}':\t{}".format(metric, value) for metric, value in sorted(self.eval_streaming_metrics_last.items())])
            tf.logging.info("Evaluation metrics: [{}]".format(eval_metrics_str))
            
            #Logs stats for time delay for the first item recommendation since its first click
            self.clicked_items_state.log_stats_time_for_first_rec()

            tf.logging.info("Restoring items state checkpoint from train")
            #Restoring the original state of items popularity and recency state from train loop
            self.clicked_items_state.restore_state_checkpoint()

    @staticmethod
    def create_eval_metrics(top_n, 
                            eval_negative_sample_relevance,
                            eval_metrics_by_session_position, 
                            content_article_embeddings_matrix,
                            articles_metadata,
                            clicked_items_state):

        relevance_positive_sample = 1.0
        #Empirical: The weight of negative samples
        relevance_negative_samples = eval_negative_sample_relevance     

        recent_clicks_buffer = clicked_items_state.get_recent_clicks_buffer()    

        eval_metrics = [metric(topn=top_n) for metric in [HitRate, MRR, NDCG]]

        eval_metrics.append(ItemCoverage(top_n, recent_clicks_buffer))  
        eval_metrics.append(ExpectedRankSensitiveNovelty(top_n)) 
        eval_metrics.append(ExpectedRankRelevanceSensitiveNovelty(top_n, relevance_positive_sample, relevance_negative_samples))
        eval_metrics.append(ContentExpectedRankRelativeSensitiveIntraListDiversity(top_n, content_article_embeddings_matrix))        
        eval_metrics.append(ContentExpectedRankRelativeRelevanceSensitiveIntraListDiversity(top_n, content_article_embeddings_matrix, relevance_positive_sample, relevance_negative_samples))
        #eval_metrics.append(CategoryExpectedIntraListDiversity(top_n, articles_metadata['category_id']))

        if eval_metrics_by_session_position:
            eval_metrics.append(HitRateBySessionPosition(top_n))

        return eval_metrics