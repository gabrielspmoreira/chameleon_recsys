from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import numpy as np
from scipy.sparse import csr_matrix
from itertools import permutations
from collections import Counter
from copy import deepcopy

from tensorflow.contrib.layers import xavier_initializer, variance_scaling_initializer
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from .metrics import HitRate, MRR
from .utils import merge_two_dicts, get_tf_dtype, hash_str_to_int
from .evaluation import compute_metrics


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

def get_tf_dtype(dtype):
        if dtype == 'int':
            tf_dtype = tf.int64
        elif dtype == 'float':
            tf_dtype = tf.float32
        #elif dtype == 'string':
        #    tf_dtype = tf.string
        else:
            raise Exception('Invalid dtype "{}"'.format(dtype))
        return tf_dtype    

class NARModuleModel():
    
    def __init__(self, mode, inputs, labels,  
                 session_features_config,
                 articles_features_config,
                 batch_size, 
                 lr, keep_prob, negative_samples, negative_sample_from_buffer,
                 content_article_embeddings_matrix,
                 rnn_num_layers=1,            
                 cosine_loss_gamma=1.0,
                 reg_weight_decay=0.0, 
                 recent_clicks_buffer_size = 1000, 
                 articles_metadata=None,
                 plot_histograms=False,
                 metrics_top_n=5,
                 elapsed_days_smooth_log_base=1.3,
                 popularity_smooth_log_base=2.0,
                 CAR_embedding_size=256,
                 rnn_units=256,
                 max_cardinality_for_ohe=30
                ):        
        
        self.lr = lr 
        self.keep_prob = keep_prob
        
        self.elapsed_days_smooth_log_base = elapsed_days_smooth_log_base
        self.popularity_smooth_log_base = popularity_smooth_log_base
        
        self.is_training = (mode == tf.estimator.ModeKeys.TRAIN)   
        
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

        with tf.variable_scope("article_content_embeddings"):
            #self.articles_metadata_columns_dict = dict([(column, id) for id, column in enumerate(articles_metadata_columns)])
            #self.articles_metadata = tf.constant(articles_metadata_values, 
            #                                     shape=articles_metadata_values.shape, 
            #                                     dtype=tf.int64)
            self.articles_metadata = {}
            #Converting Article metadata feature vectors to constants in the graph, to avoid many copies
            for feature_name in articles_metadata:
                self.articles_metadata[feature_name] = tf.constant(articles_metadata[feature_name], 
                                                 shape=articles_metadata[feature_name].shape, 
                                                 dtype=get_tf_dtype(articles_features_config[feature_name]['dtype']))


            self.items_vocab_size = articles_features_config['article_id']['cardinality']
            #self.publishers_vocab_size = articles_features_config['sequence_features']['publisher_id']['cardinality']
            #self.categories_vocab_size = articles_features_config['sequence_features']['category_id']['cardinality']

        
            self.content_article_embeddings_matrix = \
                tf.constant(content_article_embeddings_matrix, 
                            shape=content_article_embeddings_matrix.shape,
                            dtype=tf.float32)
        
        with tf.variable_scope("articles_status"):
            self.articles_pop = tf.placeholder(name="articles_pop",
                                               shape=[self.items_vocab_size],
                                               dtype=tf.int64)
            tf.summary.scalar('total_items_clicked', family='stats', tensor=tf.count_nonzero(self.articles_pop))
            
            self.articles_pop_recently_clicked = tf.placeholder(name="articles_pop_recently_clicked",
                                                               shape=[self.items_vocab_size],
                                                               dtype=tf.int64)
            tf.summary.scalar('total_items_clicked_recently', family='stats', tensor=tf.count_nonzero(self.articles_pop_recently_clicked))
            
            self.pop_recent_items_buffer = tf.placeholder(name="pop_recent_items",
                                               shape=[recent_clicks_buffer_size],
                                               dtype=tf.int64)
       

        #PS: variance_scaling_initializer() is recommended for RELU activations in https://arxiv.org/abs/1502.01852
        #whilst xavier_initializer is recommended for tanh activations
        with tf.variable_scope("main", initializer=xavier_initializer()):
            

            #Initializes CAR item embeddings variable
            self.create_item_embed_lookup_variable()
            
            
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
                self.batch_negative_items = batch_negative_items
            
            
            
            #WARNING: Must keep these variables under the same variable scope, to avoid leaking the positive item to the network (probably due to normalization)
            with tf.variable_scope("user_items_contextual_features"):
                user_context_features_concat = self.get_context_features(inputs, 
                            features_config=self.session_features_config['sequence_features'],
                            features_to_ignore=SESSION_REQ_SEQ_FEATURES)
                            
                user_context_features = tf.contrib.layers.layer_norm(user_context_features_concat, center=True, scale=True, begin_norm_axis=2)
                if self.plot_histograms:
                    tf.summary.histogram("user_context_features", user_context_features)
            

                input_items_features = self.get_item_features(item_clicked, event_timestamp, 'clicked')            
                input_user_items_features = tf.concat([user_context_features] + [input_items_features], axis=2)                
                if self.plot_histograms:
                    tf.summary.histogram("input_items_features", input_items_features)


                positive_items_features = self.get_item_features(next_item_label, max_event_timestamp, 'positive')
                if self.plot_histograms:
                    tf.summary.histogram("positive_items_features", positive_items_features)
                positive_user_items_features = tf.concat([user_context_features, positive_items_features], axis=2)


                negative_items_features = self.get_item_features(batch_negative_items, max_event_timestamp, 'negative')
                if self.plot_histograms:
                    tf.summary.histogram("negative_items_features", negative_items_features)

            #TODO: Test again batch normalization instead of layer norm (applying activation function after the normalization - dense(activation=none) + batch_norm(activation=X))
            
            with tf.variable_scope("CAR"):
                PreCAR_dense = tf.layers.Dense(512,
                                            #TODO: Test tf.nn.elu (has non-zero gradient for values < 0 and function is smooth everywhere)
                                            activation=tf.nn.leaky_relu, 
                                            #TODO: Test variance_scaling_initializer(mode="FAN_AVG"), to use the avg of fan_in and fan_out (default is just fan_in)
                                            kernel_initializer=variance_scaling_initializer(),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay),
                                            name="PreCAR_representation"
                                           )  

                input_contextual_item_embedding_pre_CAR = PreCAR_dense(input_user_items_features)
                #tf.summary.scalar('input_contextual_item_embedding_pre_CAR/fraction_of_zero_values', tf.nn.zero_fraction(input_contextual_item_embedding_pre_CAR))

                #input_contextual_item_embedding_pre_CAR_dropout = tf.layers.dropout(inputs=input_contextual_item_embedding_pre_CAR, 
                #                           rate=1.0-self.keep_prob, 
                #                           training=self.is_training)

                CAR_dense_pre_dropout = tf.layers.Dropout(rate=1.0-self.keep_prob)
                
                CAR_dense = tf.layers.Dense(CAR_embedding_size,
                                            #activation=tf.nn.relu, 
                                            activation=tf.nn.tanh, 
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay),
                                            name="CAR_representation"
                                       )
            
            with tf.variable_scope("user_personalized_contextual_article_embedding"):
                with tf.variable_scope("input"):
                    input_contextual_item_embedding = CAR_dense(CAR_dense_pre_dropout(input_contextual_item_embedding_pre_CAR))
                    #tf.summary.scalar('input_contextual_item_embedding/fraction_of_zero_values', tf.nn.zero_fraction(input_contextual_item_embedding))
          
                    if self.plot_histograms:
                        tf.summary.histogram("input_contextual_item_embedding", input_contextual_item_embedding)
                    
                with tf.variable_scope("positive"): 
                    positive_contextual_item_embedding = tf.nn.l2_normalize(CAR_dense(CAR_dense_pre_dropout(PreCAR_dense(positive_user_items_features))), axis=-1)
                    if self.plot_histograms:
                        tf.summary.histogram("positive_contextual_item_embedding", positive_contextual_item_embedding)

                with tf.variable_scope("negative"): 
                    negative_contextual_input_features =  cartesian_product(user_context_features, 
                                                                                 negative_items_features, 
                                                                                 axis=1)                            
                    #Apply l2-norm to be able to compute cosine similarity by matrix multiplication
                    negative_contextual_item_embedding = tf.nn.l2_normalize(CAR_dense(CAR_dense_pre_dropout(PreCAR_dense(negative_contextual_input_features))), axis=-1)               
                    if self.plot_histograms:
                        tf.summary.histogram("negative_contextual_item_embedding", negative_contextual_item_embedding)                            

            #Building RNN
            rnn_outputs = self.build_rnn(input_contextual_item_embedding, seq_lengths, rnn_units=rnn_units)

            #tf.summary.scalar('rnn_outputs/fraction_of_zero_values', tf.nn.zero_fraction(input_contextual_item_embedding_pre_CAR))
  
            with tf.variable_scope("session_representation"): 
                rnn_outputs_fc1 = tf.layers.dense(rnn_outputs, 512,                                        
                                            #TODO: Test tf.nn.elu (has non-zero gradient for values < 0 and function is smooth everywhere)
                                            activation=tf.nn.leaky_relu, 
                                            kernel_initializer=variance_scaling_initializer(),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay),
                                            name="FC1"
                                           )

                #tf.summary.scalar('rnn_outputs_fc1/fraction_of_zero_values', tf.nn.zero_fraction(rnn_outputs_fc1))

                rnn_outputs_fc1_dropout = tf.layers.dropout(inputs=rnn_outputs_fc1, 
                                           rate=1.0-self.keep_prob, 
                                           training=self.is_training)
                
                
                rnn_outputs_fc2 = tf.layers.dense(rnn_outputs_fc1_dropout, CAR_embedding_size,
                    #activation=tf.nn.relu, 
                    activation=tf.nn.tanh, 
                    name='FC2', 
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay))

                #tf.summary.scalar('rnn_outputs_fc2/fraction_of_zero_values', tf.nn.zero_fraction(rnn_outputs_fc1))
                
                if self.plot_histograms:
                    tf.summary.histogram("rnn_outputs_fc2", rnn_outputs_fc2)
                
            
            with tf.variable_scope("predicted_contextual_item_embedding"): 
                #Continuing with DSSM losss
                #Apply l2-norm to be able to compute cosine similarity by matrix multiplication
                predicted_contextual_item_embedding = tf.nn.l2_normalize(rnn_outputs_fc2, axis=-1)
                
                if self.plot_histograms:
                    tf.summary.histogram("predicted_contextual_item_embedding", predicted_contextual_item_embedding)

            with tf.variable_scope("recommendations_ranking"): 
                with tf.variable_scope("cos_sim_positive"):
                    #Computing Cosine similarity between predicted embedding and positive embedding (label)
                    cos_sim_positive = tf.reduce_sum(tf.multiply(positive_contextual_item_embedding, 
                                                                 predicted_contextual_item_embedding), 
                                                     axis=-1, keepdims=True)
                    #print("cos_sim_positive", cos_sim_positive.shape)
                    if self.plot_histograms:
                        tf.summary.histogram("train/cos_sim_positive", cos_sim_positive)
                    
                    
                with tf.variable_scope("cos_sim_negative"):
                    #Computing Cosine similarity between predicted embedding and negative items embedding
                    cos_sim_negative = tf.reduce_sum(tf.multiply(negative_contextual_item_embedding, 
                                                     tf.expand_dims(predicted_contextual_item_embedding, 2)), axis=-1)
                    #print("cos_sim_negative", cos_sim_negative.shape)
                    if self.plot_histograms:
                        tf.summary.histogram("train/cos_sim_negative", cos_sim_negative)
     
                
                with tf.variable_scope("positive_prob"):
                    gamma_var = tf.get_variable('gamma', dtype=tf.float32, trainable=True, 
                                                initializer=tf.constant(cosine_loss_gamma))
                    tf.summary.scalar('gamma', family='train', tensor=gamma_var)

                    #Concatenating cosine similarities (positive + K sampled negative)
                    cos_sim_concat = tf.concat([cos_sim_positive, cos_sim_negative], axis=2)
                    cos_sim_concat_scaled = cos_sim_concat * gamma_var
                    #Computing softmax over cosine similarities
                    items_prob = tf.nn.softmax(cos_sim_concat_scaled) 
                
                
                if mode == tf.estimator.ModeKeys.EVAL:
                    #Computing evaluation metrics
                    self.define_eval_metrics(next_item_label, batch_negative_items, items_prob)


                with tf.variable_scope("loss"):
                    #Computing the probability of the positive item (label)
                    positive_prob = items_prob[:,:,0]
                    negative_probs = items_prob[:,:,1:]
                    #Summary of first element of the batch sequence (because others might be masked)
                    if self.plot_histograms:
                        tf.summary.histogram("positive_prob", positive_prob[:,0])
                        tf.summary.histogram("negative_probs", negative_probs[:,0,:])


                    #Computing batch loss
                    loss_mask = tf.to_float(self.item_clicked_mask)
                    masked_loss = tf.multiply(tf.log(positive_prob), loss_mask)
                     
                    #Averaging the loss by the number of masked items in the batch
                    cosine_sim_loss = -tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask) 
                    tf.summary.scalar("train/cosine_sim_loss", family='train', tensor=cosine_sim_loss)
                    
                    #reg_loss = self.reg_weight_decay * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if not ("noreg" in tf_var.name or "Bias" in tf_var.name))
                    reg_loss = tf.losses.get_regularization_loss()
                    tf.summary.scalar("train/reg_loss", family='train', tensor=reg_loss)
                    
                    self.total_loss = cosine_sim_loss  + reg_loss
                    tf.summary.scalar("train/total_loss", family='train', tensor=self.total_loss)

            
           
                     
            if mode == tf.estimator.ModeKeys.TRAIN:
                with tf.variable_scope('training'):
                    opt = tf.train.AdamOptimizer(self.lr,
                                                 beta1=0.9,
                                                 beta2=0.999,
                                                 epsilon=1e-08)


                    #Necessary to run update ops for batch_norm, streaming metrics
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)                
                    with tf.control_dependencies(update_ops):      

                        #self.train = opt.minimize(self.total_loss, global_step=self.gs)
                        # Get the gradient pairs (Tensor, Variable)
                        grads = opt.compute_gradients(self.total_loss)
                        # Update the weights wrt to the gradient
                        self.train = opt.apply_gradients(grads, 
                                                         global_step=tf.train.get_global_step()#self.gs
                                                        )


                        if self.plot_histograms:
                            # Save the grads with tf.summary.histogram (only for debug - SLOW!)
                            for index, grad in enumerate(grads):
                                try:
                                    tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])
                                except Exception as e:
                                    print("ERROR generating histogram for %d - %s: %s" % (index, grads[index][1].name, e))
            

    def get_context_features(self, inputs, features_config, 
                             features_to_ignore):

        def cat_ohe(feature_name, size, inputs):
            return tf.one_hot(inputs[feature_name], size, name="{}_cat_one_hot".format(feature_name))
                
        def cat_embed(feature_name, size, inputs):
            #print("GET_CONTEXT_FEATURES(): {}_cat_embedding".format(feature_name))

            with tf.variable_scope("{}_cat_embedding".format(feature_name), reuse=tf.AUTO_REUSE):        
                dim =  get_embedding_size(size)
                embeddings = tf.get_variable("{}_embedding".format(feature_name), shape=[size, dim],
                                             regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay))
                lookup = tf.nn.embedding_lookup(embeddings, ids=inputs[feature_name])#, max_norm=1)
                return lookup

        with tf.variable_scope("context_features"):
            context_features_list = []
            for feature_name in features_config:
                #Ignores item_clicked and timestamp as user contextual features
                if feature_name in features_to_ignore:
                    continue

                if features_config[feature_name]['type'] == 'categorical':
                    size = features_config[feature_name]['cardinality']
                    if features_config[feature_name]['cardinality'] < self.max_cardinality_for_ohe:
                        feature_op = cat_ohe(feature_name, size, inputs)
                    else:
                        feature_op = cat_embed(feature_name, size, inputs)
                elif features_config[feature_name]['type'] == 'numerical':
                    feature_op = tf.expand_dims(inputs[feature_name], -1)
                else:
                    raise Exception('Invalid feature type: {}'.format(feature_name))
                context_features_list.append(feature_op)

            context_features_concat = tf.concat(context_features_list, axis=2)
            return context_features_concat




            
    def define_eval_metrics(self, next_item_label, batch_negative_items, items_prob):
        with tf.variable_scope("evaluation_metrics"):
            
            with tf.variable_scope("predicted_items"):

                next_item_label_expanded = tf.expand_dims(next_item_label, -1)
                batch_negative_items_tiled = tf.tile(tf.expand_dims(batch_negative_items, 1), [1, tf.shape(next_item_label_expanded)[1], 1])

                pos_neg_items_concat = tf.concat([next_item_label_expanded, batch_negative_items_tiled], 2)

                #Predicting item ids from [positive + k negative samples]
                items_top_prob_indexes = tf.nn.top_k(items_prob,  k=tf.shape(items_prob)[2]).indices

                #In older versions of TF
                #items_top_prob_indexes_idx = array_ops.where(
                #        math_ops.not_equal(items_top_prob_indexes, tf.constant(-1, tf.int32)))
                items_top_prob_indexes_idx = tf.contrib.layers.dense_to_sparse(items_top_prob_indexes, eos_token=-1).indices
                
                items_top_prob_indexes_val = tf.gather_nd(items_top_prob_indexes, items_top_prob_indexes_idx)
                #Takes the first two columns of the index and use sorted indices as the last column
                items_top_prob_reordered_indexes = tf.concat([items_top_prob_indexes_idx[:,:2], 
                                                              tf.expand_dims(tf.cast(items_top_prob_indexes_val, tf.int64), 1)], 1)
                predicted_item_ids = tf.reshape(tf.gather_nd(pos_neg_items_concat, items_top_prob_reordered_indexes), 
                                                tf.shape(pos_neg_items_concat))
                self.predicted_item_ids = predicted_item_ids

            #Computing Accuracy@1
            self.next_item_accuracy_at_1, self.next_item_accuracy_at_1_update_op = \
                                tf.metrics.accuracy(predictions=predicted_item_ids[:,:,0], 
                                                    labels=next_item_label, 
                                                    weights=tf.to_float(self.item_clicked_mask),
                                                    name='accuracy_at_1')



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
            self.ndcg_at_n_mean, self.ndcg_at_n_mean_update_op = \
                self.define_ndcg_metric(predicted_item_ids, next_item_label_expanded, topk=self.metrics_top_n)


    def define_ndcg_metric(self, predicted_item_ids, next_item_label_expanded, topk):
        with tf.variable_scope("ndcg"):
            #Computing NDCG
            predicted_correct = tf.to_int32(tf.equal(predicted_item_ids, next_item_label_expanded))
            ndcg_predicted = tf_ndcg_at_k(predicted_correct, topk)
            
            #Combining masks of padding items and NDCG zeroed values (because the correct value is not in the top n)
            #ndcg_mask = tf.multiply(tf.to_float(self.item_clicked_mask), tf.to_float(tf.sign(ndcg_predicted)))
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

            
    def get_layer_norm_item_features(self, item_features):
        with tf.variable_scope("layer_norm_item_features", reuse=tf.AUTO_REUSE):
            item_features_scaled = tf.contrib.layers.layer_norm(item_features, center=True, scale=True, begin_norm_axis=2)
            
            return item_features_scaled
        
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
            #items_ohe = tf.one_hot(item_ids, self.items_vocab_size)

            item_clicked_interactions_embedding = self.items_cat_embed(item_ids)


            items_acr_embeddings_lookup = tf.nn.embedding_lookup(self.content_embedding_variable, ids=item_ids)

            
            #Obtaining item features for specified items (e.g. clicked, negative samples)
            item_contextual_features = {}
            for feature_name in self.articles_features_config:
                if feature_name not in ARTICLE_REQ_FEATURES:
                    item_contextual_features[feature_name] = \
                            tf.gather(self.articles_metadata[feature_name], item_ids)

            #Concatenating item contextual features
            item_contextual_features = self.get_context_features(item_contextual_features, 
                            features_config=self.articles_features_config, 
                            features_to_ignore=ARTICLE_REQ_FEATURES)


            #Taking the maximum timestamp of the batch
            #max_event_timestamp = tf.reduce_max(event_timestamp)
            #Computing Item Dynamic features (RECENCY and POPULARITY)
            items_dynamic_features = self.get_items_dynamic_features(item_ids, 
                                                        events_timestamp, 
                                                        summary_suffix=summary_suffix)

            #Creating a feature specifically to inform the network whether this is a padding item or not
            #item_clicked_not_padding = tf.expand_dims(tf.cast(tf.sign(item_ids), tf.float32), axis=-1)

            items_features_list = [ 
                                    #Item embedding trained by ACR module
                                    items_acr_embeddings_lookup, 
                                    #Trainable item embedding
                                    item_clicked_interactions_embedding,
                                    item_contextual_features,
                                    items_dynamic_features, 
                                    #item_clicked_not_padding
                                    ]


            items_features_concat = tf.concat(items_features_list, axis=2)


            #tf.summary.histogram("items_features_norm_BEFORE", tf.boolean_mask(input_items_features_concat, self.item_clicked_mask))
            items_features_norm = self.get_layer_norm_item_features(items_features_concat)
            #tf.summary.histogram("items_features_norm_AFTER", tf.boolean_mask(input_items_features, self.item_clicked_mask))

            return items_features_norm
       
    def normalize_values(self, tensor_to_normalize, tensor_to_get_stats_from): 
        with tf.variable_scope("values_normalization"):       
            mean, variance  = tf.nn.moments(tensor_to_get_stats_from, axes=[0])        
            
            #Fixing size of stats to avoid dynamic last dimension on tensor_normed
            mean = tf.reshape(mean, [1])
            variance = tf.reshape(variance, [1])
             
            stddev = tf.sqrt(variance)
            
            #To avoid division by zero
            epsilon = tf.constant(1e-8)        
            tensor_normed = (tensor_to_normalize - mean)  / (stddev + epsilon)
            return tensor_normed    
    
    def get_unique_items_from_pop_recent_buffer(self):
        with tf.variable_scope("unique_items_from_pop_recent_buffer"):    
            recent_items_unique, _ = tf.unique(self.pop_recent_items_buffer)
            #Removing zero
            recent_items_unique = tf.boolean_mask(recent_items_unique,
                                                  tf.cast(tf.sign(recent_items_unique), tf.bool)) 
            return recent_items_unique
    
    def calculate_items_recency(self, creation_dates, reference_timestamps):
        with tf.variable_scope("calculate_items_recency"):    
            elapsed_days = tf.nn.relu(((tf.to_float(reference_timestamps) / tf.constant(1000.0)) \
                               - tf.to_float(creation_dates)) / tf.constant(60.0 * 60.0 * 24.0))
            
            elapsed_days_smoothed = log_1p(elapsed_days, base=self.elapsed_days_smooth_log_base)
            
            return elapsed_days_smoothed
    
    def normalize_recency_feature(self, batch_elapsed_days_since_publishing, batch_events_timestamp, item_ids):
        with tf.variable_scope("normalize_recency_feature"):   
            #Computing global recency stats from buffer
            recent_items_unique = self.get_unique_items_from_pop_recent_buffer()
            recent_items_creation_date = tf.gather(self.articles_metadata['created_at_ts'], recent_items_unique)
            recent_items_elapsed_days_since_creation = self.calculate_items_recency(recent_items_creation_date, 
                                                                                                      tf.reduce_max(batch_events_timestamp))
            recent_items_elapsed_days_since_creation_smoothed = log_1p(recent_items_elapsed_days_since_creation, 
                                                                       base=self.elapsed_days_smooth_log_base)
            

            #Normalizing batch recency feature
            batch_elapsed_days_since_publishing_smoothed = log_1p(batch_elapsed_days_since_publishing, 
                                                                  base=self.elapsed_days_smooth_log_base)
            
            #If there aren't recent items available in the buffer (first batch), use batch items to compute norm stats
            tensor_to_get_stats_from = tf.cond(tf.equal(tf.shape(recent_items_elapsed_days_since_creation_smoothed)[0], tf.constant(0)), 
                                               lambda: tf.boolean_mask(batch_elapsed_days_since_publishing_smoothed, tf.cast(tf.sign(item_ids), tf.bool)),
                                               lambda: recent_items_elapsed_days_since_creation_smoothed)
                                               
            batch_elapsed_days_since_publishing_normed = self.normalize_values(batch_elapsed_days_since_publishing_smoothed, 
                                                                               tensor_to_get_stats_from)
            return batch_elapsed_days_since_publishing_normed
        
    
    def get_items_recency_feature(self, item_ids, events_timestamp, summary_suffix=''):
        with tf.variable_scope("items_recency_feature"):  
            #Computing RECENCY feature
            batch_articles_creation_date = tf.gather(tf.reshape(self.articles_metadata['created_at_ts'], 
                                                                [-1,1]), item_ids)
            elapsed_days_since_publishing = self.calculate_items_recency(batch_articles_creation_date, events_timestamp)
            if self.plot_histograms:
                tf.summary.histogram('batch_elapsed_days_since_publishing/'+summary_suffix, family='stats',
                                  values=tf.boolean_mask(elapsed_days_since_publishing, tf.cast(tf.sign(item_ids), tf.bool)))
            
            
            elapsed_days_since_publishing_norm = self.normalize_recency_feature(elapsed_days_since_publishing, 
                                                                                  events_timestamp, item_ids)

            if self.plot_histograms:
                tf.summary.histogram('batch_elapsed_days_since_publishing_norm/'+summary_suffix, family='stats',
                                  values=tf.boolean_mask(elapsed_days_since_publishing_norm, tf.cast(tf.sign(item_ids), tf.bool)))
                    
            
            return elapsed_days_since_publishing_norm, batch_articles_creation_date
    
    
    def normalize_popularity_feature(self, batch_items_pop, item_ids):
        with tf.variable_scope("popularity_feature_normalization"): 
            #Computing global recency stats from buffer
            recent_items_unique = self.get_unique_items_from_pop_recent_buffer()
            recent_items_pop = tf.gather(self.articles_pop_recently_clicked, recent_items_unique)
            recent_items_pop_smoothed = log_1p(recent_items_pop, 
                                               base=self.popularity_smooth_log_base)
            self.recent_items_pop_smoothed = recent_items_pop_smoothed
            
            #Normalizing batch recency feature
            batch_items_pop_smoothed = log_1p(batch_items_pop, 
                                              base=self.popularity_smooth_log_base)
            
            #If there aren't recent items available in the buffer (first batch), use batch items to compute norm stats
            tensor_to_get_stats_from = tf.cond(tf.equal(tf.shape(recent_items_pop_smoothed)[0], tf.constant(0)), 
                                               lambda: tf.boolean_mask(batch_items_pop_smoothed, 
                                                               tf.cast(tf.sign(item_ids), tf.bool)),
                                               lambda: recent_items_pop_smoothed)
            
            batch_items_pop_normed = self.normalize_values(batch_items_pop_smoothed, 
                                                        tensor_to_get_stats_from)
            
            return batch_items_pop_normed
    
    def get_items_popularity_feature(self, item_ids, summary_suffix=''):
        #Computing POPULARITY feature
        with tf.variable_scope("items_popularity_feature"):             
            
            #batch_articles_pop = tf.to_float(tf.gather(self.articles_pop, tf.expand_dims(item_ids, -1)))                
            batch_articles_pop = tf.to_float(tf.gather(self.articles_pop_recently_clicked, tf.expand_dims(item_ids, -1)))               

            if self.plot_histograms:
                tf.summary.histogram('batch_articles_pop/'+summary_suffix, family='stats',
                                  values=tf.boolean_mask(batch_articles_pop, tf.cast(tf.sign(item_ids), tf.bool)))
            
            batch_articles_pop_norm = self.normalize_popularity_feature(batch_articles_pop, item_ids)

            if self.plot_histograms:
                tf.summary.histogram('batch_articles_pop_norm/'+summary_suffix, family='stats',
                                 values=tf.boolean_mask(batch_articles_pop_norm, tf.cast(tf.sign(item_ids), tf.bool)))
            
            return batch_articles_pop_norm

                                 
    
    def get_items_dynamic_features(self, item_ids, events_timestamp, summary_suffix=''):
        
        with tf.variable_scope("items_dynamic_features", reuse=tf.AUTO_REUSE):
        
            #Computing RECENCY feature
            elapsed_days_since_publishing_log, batch_articles_creation_date = \
                        self.get_items_recency_feature(item_ids, events_timestamp, summary_suffix=summary_suffix)

            #Computing POPULARITY feature          
            batch_articles_pop_log = self.get_items_popularity_feature(item_ids,
                                                                       summary_suffix=summary_suffix) 



            dynamic_features_concat = tf.concat([elapsed_days_since_publishing_log,
                                                 batch_articles_pop_log],
                                                axis=2)

        return dynamic_features_concat
            
    def get_sample_from_recently_clicked_items_buffer(self, sample_size):
        with tf.variable_scope("neg_samples_buffer"):
            pop_recent_items_buffer_masked = tf.boolean_mask(self.pop_recent_items_buffer,
                                                      tf.cast(tf.sign(self.pop_recent_items_buffer), tf.bool)) 
            
            unique_pop_recent_items_buffer_masked, _ = tf.unique(pop_recent_items_buffer_masked)
            #tf.summary.scalar('unique_clicked_items_on_buffer', family='stats', tensor=tf.shape(unique_pop_recent_items_buffer_masked)[0])
            tf.summary.scalar('clicked_items_on_buffer', family='stats', tensor=tf.shape(pop_recent_items_buffer_masked)[0])
            
            #recent_items_unique_sample, idxs = tf.unique(tf.random_shuffle(pop_recent_items_buffer_masked)[:sample_size*sample_size_factor_to_look_for_unique])
            recent_items_unique_sample = tf.random_shuffle(unique_pop_recent_items_buffer_masked)
            
            #Samples K articles from recent articles
            #sample_recent_articles_ids = tf.random_shuffle(articles_metadata_creation_date_past_only)[:recent_articles_samples_for_eval][:,self.articles_metadata_columns_dict['article_id']]
            sample_recently_clicked_items = recent_items_unique_sample[:sample_size]
            return sample_recently_clicked_items
    

    def get_masked_seq_values(self, tensor):
        return tf.boolean_mask(tensor, self.item_clicked_mask, name='masked_values')      

    
    def get_negative_samples(self, item_clicked, candidate_samples):  
        with tf.variable_scope("negative_samples"):      
            current_batch_size = tf.shape(item_clicked)[0]
            
            #Repeating all unique items for each sample in the batch
            batch_candidate_negative_items = tf.reshape(tf.tile(candidate_samples, [self.batch_size]), [self.batch_size,-1])
            #Reducing rows if batch size is lower than the default (last step)
            batch_candidate_negative_items = batch_candidate_negative_items[:current_batch_size,:]        
            #For each batch sample, filters out session items to keep only negative items 
            #Ps. remove last columns (according to max session size) to remove padding zeros. 
            #    Side effect is that higher item ids are ignored for shorter sessions (because set_difference() sorts ids increasinly)
            batch_negative_items = tf.sparse_tensor_to_dense(tf.sets.set_difference(batch_candidate_negative_items, 
                                                                                    item_clicked))
            return batch_negative_items
         
    def get_batch_negative_samples(self, item_clicked, additional_samples, num_negative_samples):
        with tf.variable_scope("neg_samples_batch"):
            current_batch_size, batch_max_session_length = tf.shape(item_clicked)[0], tf.shape(item_clicked)[1] 

            batch_items = tf.reshape(item_clicked, [-1])
            #Removing padded (zeroed) items
            batch_items_unique, _ = tf.unique(tf.boolean_mask(batch_items, tf.cast(tf.sign(batch_items), dtype=tf.bool)))
                            
            #Concatenating batch items with additional samples (to deal with small batches)
            candidate_neg_items = tf.concat([batch_items_unique, additional_samples], axis=0)        
            
            #Ignoring zeroes in the end of neg. samples matrix
            batch_negative_items = self.get_negative_samples(item_clicked, candidate_neg_items) \
                                   [:, :-tf.maximum(1,batch_max_session_length-1)]         

            #Randomly picks K negative samples for each batch sample 
            #Ps. transpose() is necessary because random_shuffle() only shuffles first dimension, and we want to shuffle the second dimension
            #batch_negative_items = tf.transpose(tf.random_shuffle(tf.transpose(batch_negative_items)))[:,:num_negative_samples]
            
            batch_negative_items = shuffle_columns(batch_negative_items)[:,:num_negative_samples]
            
            return batch_negative_items

            
    def build_rnn(self, the_input, lengths, rnn_units=256):    
        with tf.variable_scope("RNN"):    
            fw_cells = []
            for _ in range(self.rnn_num_layers):
                #cell = tf.nn.rnn_cell.GRUCell(rnn_units)  
                cell = tf.nn.rnn_cell.LSTMCell(rnn_units, state_is_tuple=True)              
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
                                                     output_keep_prob=self.keep_prob, 
                                                     input_keep_prob=self.keep_prob)
                fw_cells.append(cell)   
            
            fw_stacked_cells = tf.contrib.rnn.MultiRNNCell(fw_cells, state_is_tuple=True)
            rnn_outputs, rnn_final_hidden_state_tuples = \
                tf.nn.dynamic_rnn(fw_stacked_cells, the_input, dtype=tf.float32, sequence_length=lengths)

            if self.plot_histograms:
                tf.summary.histogram("rnn/outputs", rnn_outputs)       
            
            return rnn_outputs
    
    def create_item_embed_lookup_variable(self):        
        with tf.variable_scope("item_embedding"):          
            self.content_embedding_variable = tf.Variable(self.content_article_embeddings_matrix,
                                                          trainable=False)








class ClickedItemsState:
    
    def __init__(self, recent_clicks_buffer_size, num_items):
        self.recent_clicks_buffer_size = recent_clicks_buffer_size
        self.num_items = num_items           
        self.reset_state()
        
    def reset_state(self):
        #Global state
        self.articles_pop = np.zeros(shape=[self.num_items], dtype=np.int64)
        self.pop_recent_clicks_buffer = np.zeros(shape=[self.recent_clicks_buffer_size], dtype=np.int64)
        #State shared by ItemCooccurrenceRecommender and ItemKNNRecommender
        self.items_coocurrences = csr_matrix((self.num_items, self.num_items), dtype=np.int64)
        #States specific for benchmarks
        self.benchmarks_states = dict()
        
    def save_state_checkpoint(self):
        self.articles_pop_chkp = np.copy(self.articles_pop)
        self.pop_recent_clicks_buffer_chkp = np.copy(self.pop_recent_clicks_buffer)
        self.items_coocurrences_chkp = csr_matrix.copy(self.items_coocurrences)
        self.benchmarks_states_chkp = deepcopy(self.benchmarks_states)
        
    def restore_state_checkpoint(self):
        self.articles_pop = self.articles_pop_chkp
        del self.articles_pop_chkp
        self.pop_recent_clicks_buffer = self.pop_recent_clicks_buffer_chkp
        del self.pop_recent_clicks_buffer_chkp
        self.items_coocurrences = self.items_coocurrences_chkp
        del self.items_coocurrences_chkp
        self.benchmarks_states = self.benchmarks_states_chkp
        del self.benchmarks_states_chkp
        
    def get_articles_pop(self):
        return self.articles_pop
    
    def get_recent_clicks_buffer(self):
        return self.pop_recent_clicks_buffer
    
    def get_articles_pop_from_recent_clicks_buffer(self):
        recent_clicks_buffer_nonzero = self.pop_recent_clicks_buffer[np.nonzero(self.pop_recent_clicks_buffer)]
        recent_clicks_item_counter = Counter(recent_clicks_buffer_nonzero)
        
        pop_recently_clicked = np.zeros(shape=[self.num_items], dtype=np.int64)
        pop_recently_clicked[list(recent_clicks_item_counter.keys())] = list(recent_clicks_item_counter.values())
                
        return pop_recently_clicked
    
    def get_items_coocurrences(self):
        return self.items_coocurrences
    
    def _get_non_zero_items_vector(self, batch_clicked_items):
        #Converting batch items to a vector sorted by last clicked items in sessions
        batch_items_vector = batch_clicked_items.T.reshape(-1)[::-1]
        return batch_items_vector[np.nonzero(batch_items_vector)]
    
    def update_items_state(self, batch_clicked_items):
        batch_items_nonzero = self._get_non_zero_items_vector(batch_clicked_items)
        self._update_recently_clicked_items_buffer(batch_items_nonzero)
        self._update_pop_items(batch_items_nonzero)        
            
    def _update_recently_clicked_items_buffer(self, batch_items_nonzero):
        #TODO: Keep on buffer based on time (e.g. last X hours), and not on last N clicks
        #Updating buffer with latest clicked elements
        self.pop_recent_clicks_buffer = np.hstack([batch_items_nonzero, self.pop_recent_clicks_buffer])[:self.recent_clicks_buffer_size]
        
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
                 eval_benchmark_classifiers=[],
                 eval_metrics_by_session_position=False):

        self.mode = mode
        self.model = model        
        self.eval_metrics_top_n = eval_metrics_top_n
                
        self.clicked_items_state = clicked_items_state
        self.eval_sessions_metrics_log = eval_sessions_metrics_log
        self.sessions_negative_items_log = sessions_negative_items_log


        self.bench_classifiers = [clf['recommender'](self.clicked_items_state,
                                                     clf['params'],
                                                     ItemsStateUpdaterHook.create_eval_metrics(self.eval_metrics_top_n)) for clf in eval_benchmark_classifiers]
        self.eval_metrics_by_session_position = eval_metrics_by_session_position

    def begin(self):        
        if self.mode == tf.estimator.ModeKeys.EVAL:
            tf.logging.info("Saving items state checkpoint from train")
            #Save state of items popularity and recency from train loop, to restore after evaluation finishes
            self.clicked_items_state.save_state_checkpoint()  
            
            #Resets streaming metrics
            self.eval_streaming_metrics_last = {}            
            for clf in self.bench_classifiers:
                clf.reset_eval_metrics()

            self.streaming_metrics = ItemsStateUpdaterHook.create_eval_metrics(self.eval_metrics_top_n)
            #self.metrics_by_session_pos = StreamingMetrics(topn=self.metrics_top_n)
                
            self.stats_logs = []


    #Runs before every batch
    def before_run(self, run_context): 
        fetches = {'clicked_items': self.model.item_clicked,
                   'next_item_labels': self.model.next_item_label,
                   'last_item_label': self.model.label_last_item,
                   'session_id': self.model.session_id,
                   'session_start': self.model.session_start,
                   'user_id': self.model.user_id,
                   }

        
        if self.mode == tf.estimator.ModeKeys.EVAL:
            fetches['eval_batch_negative_items'] = self.model.batch_negative_items
            fetches['batch_items_count'] = self.model.batch_items_count
            fetches['batch_unique_items_count'] = self.model.batch_unique_items_count
            
            fetches['hitrate_at_1'] = self.model.next_item_accuracy_at_1_update_op
            fetches['hitrate_at_n'] = self.model.recall_at_n_update_op
            fetches['mrr_at_n'] = self.model.mrr_update_op
            #fetches['ndcg_at_n'] = self.model.ndcg_at_n_mean_update_op
            
            fetches['predicted_item_ids'] = self.model.predicted_item_ids

            
            

        
        feed_dict = {
            self.model.articles_pop: self.clicked_items_state.get_articles_pop(),
            self.model.pop_recent_items_buffer: self.clicked_items_state.get_recent_clicks_buffer(),
            self.model.articles_pop_recently_clicked: self.clicked_items_state.get_articles_pop_from_recent_clicks_buffer()
        }               

        return tf.train.SessionRunArgs(fetches=fetches,
                                       feed_dict=feed_dict)
    
    
    def evaluate_and_update_streaming_metrics_last(self, clf, users_ids, clicked_items, next_item_labels, eval_negative_items):
        clf_metrics = clf.evaluate(users_ids, clicked_items, next_item_labels, topk=self.eval_metrics_top_n, 
                                   eval_negative_items=eval_negative_items)
        self.eval_streaming_metrics_last = merge_two_dicts(self.eval_streaming_metrics_last, clf_metrics)


    def evaluate_metrics_by_session_pos(self, predictions, labels):
        recall_by_session_pos, recall_total_by_session_pos = self.metrics_by_session_pos.recall_at_n_by_session_pos(predictions, labels, self.metrics_top_n)
        recall_by_session_pos_dict = dict([("recall_by_session_pos_{0:02d}".format(key), recall_by_session_pos[key]) for key in recall_by_session_pos])
        sessions_length_dict = dict([("sessions_length_count_{0:02d}".format(key), recall_total_by_session_pos[key]) for key in recall_total_by_session_pos])
        self.eval_streaming_metrics_last = merge_two_dicts(merge_two_dicts(self.eval_streaming_metrics_last, recall_by_session_pos_dict), sessions_length_dict)

    #Runs after every batch
    def after_run(self, run_context, run_values):     
        clicked_items = run_values.results['clicked_items']
        next_item_labels = run_values.results['next_item_labels']
        last_item_label = run_values.results['last_item_label'] 

        users_ids = run_values.results['user_id']
        sessions_ids = run_values.results['session_id']

                
        if self.mode == tf.estimator.ModeKeys.EVAL:
            self.eval_streaming_metrics_last = {}
            self.eval_streaming_metrics_last['hitrate_at_1'] = run_values.results['hitrate_at_1']
            self.eval_streaming_metrics_last['hitrate_at_n'] = run_values.results['hitrate_at_n']
            self.eval_streaming_metrics_last['mrr_at_n'] = run_values.results['mrr_at_n']
            #self.eval_streaming_metrics_last['ndcg_at_n'] = run_values.results['ndcg_at_n']


            predicted_item_ids = run_values.results['predicted_item_ids']
            #tf.logging.info('predicted_item_ids: {}'.format(predicted_item_ids))

            if self.eval_metrics_by_session_position:
                self.evaluate_metrics_by_session_pos(predicted_item_ids, next_item_labels)
            
            
            eval_batch_negative_items = run_values.results['eval_batch_negative_items']

            if self.sessions_negative_items_log != None:
                #Acumulating session negative items, to allow evaluation comparison
                # with benchmarks outsite the framework (e.g. Matrix Factorization) 
                for session_id, neg_items in zip(sessions_ids,
                                                 eval_batch_negative_items):                    
                    self.sessions_negative_items_log.append({'session_id': str(session_id), #Convert numeric session_id to str because large ints are not serializable
                                                         'negative_items': neg_items})

            batch_stats = {'eval_sampled_negative_items': eval_batch_negative_items.shape[1],
                           'batch_items_count': run_values.results['batch_items_count'],
                           'batch_unique_items_count': run_values.results['batch_unique_items_count'],
                           'batch_sessions_count': len(sessions_ids)
                           #'recent_items_buffer_filled': np.count_nonzero(clicked_items_state.get_recent_clicks_buffer()),
                          }
            self.stats_logs.append(batch_stats)
            tf.logging.info('batch_stats: {}'.format(batch_stats))


            #Computing metrics for this neural model
            model_metrics_values = compute_metrics(predicted_item_ids, next_item_labels, 
                                            self.streaming_metrics, 
                                            metrics_suffix='main')            
            self.eval_streaming_metrics_last = merge_two_dicts(self.eval_streaming_metrics_last, 
                                                               model_metrics_values)
            
            #Computing metrics for Benchmark recommenders
            for clf in self.bench_classifiers:
                tf.logging.info('Evaluating benchmark: {}'.format(clf.get_description()))    
                self.evaluate_and_update_streaming_metrics_last(clf, users_ids, 
                                clicked_items, next_item_labels, eval_batch_negative_items)
            tf.logging.info('Finished benchmarks evaluation')

        #Training benchmark classifier
        for clf in self.bench_classifiers:
            #As for GCom dataset session_ids are not timestamps, generating artificial session_ids 
            # by concatenating session_start with hashed session ids to make it straightforward to sort them by time
            #TODO: In the next generation of Gcom dataset, make this transformation before saving to TFRecord and remove from here
            '''
            sessions_ids_hashed = list([int('{}{}'.format(session_start, hash_str_to_int(session_id, 3))) \
                             for session_start, session_id in zip(run_values.results['session_start'], 
                                                                  run_values.results['session_id'])])
            clf.train(users_ids, sessions_ids_hashed, clicked_items, next_item_labels)
            '''
            clf.train(users_ids, sessions_ids, clicked_items, next_item_labels)
        
        #Concatenating all clicked items in the batch (including last label)
        batch_clicked_items = np.concatenate([clicked_items,last_item_label], axis=1)
        #Updating items state
        self.clicked_items_state.update_items_state(batch_clicked_items)        
        self.clicked_items_state.update_items_coocurrences(batch_clicked_items)
 
    
    def end(self, session=None):
        if self.mode == tf.estimator.ModeKeys.EVAL:    
            avg_neg_items = np.mean([x['eval_sampled_negative_items'] for x in self.stats_logs])
            self.eval_streaming_metrics_last['avg_eval_sampled_neg_items'] = avg_neg_items
            
            clicks_count = np.sum([x['batch_items_count'] for x in self.stats_logs])
            self.eval_streaming_metrics_last['clicks_count'] = clicks_count

            sessions_count = np.sum([x['batch_sessions_count'] for x in self.stats_logs])
            self.eval_streaming_metrics_last['sessions_count'] = sessions_count
                        
            self.eval_sessions_metrics_log.append(self.eval_streaming_metrics_last)
            eval_metrics_str = '\n'.join(["'{}':\t{:.4f}".format(metric, value) for metric, value in sorted(self.eval_streaming_metrics_last.items())])
            tf.logging.info("Evaluation metrics: [{}]".format(eval_metrics_str))
            
            tf.logging.info("Restoring items state checkpoint from train")
            #Restoring the original state of items popularity and recency state from train loop
            self.clicked_items_state.restore_state_checkpoint()

    @staticmethod
    def create_eval_metrics(top_n):
        eval_metrics = [metric(topn=top_n) for metric in [HitRate, MRR]]
        return eval_metrics