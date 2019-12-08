#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/9/23 2:52
# @Author : {ZM7}
# @File : utils.py
# @Software: PyCharm

#Adapted from https://github.com/CRIPAC-DIG/SR-GNN/tree/master/tensorflow_code and from https://github.com/rn5l/session-rec
#to work as a baseline for CHAMELEON


import tensorflow as tf
import math
from .utils import Data, prepare_data
import numpy as np
import datetime
import pandas as pd

from ...evaluation import update_metrics, compute_metrics_results
from ...metrics import HitRate, MRR

class Model(object):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, nonhybrid=True):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        
        self.nonhybrid = nonhybrid
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

    def forward(self, re_embedding, batch_size, train=True):
        rm = tf.reduce_sum(self.mask, 1)
        last_id = tf.gather_nd(self.alias, tf.stack([tf.range(batch_size), tf.to_int32(rm)-1], axis=1))
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(batch_size), last_id], axis=1))
        seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], self.alias[i]) for i in range(batch_size)],
                         axis=0)                                                           #batch_size*T*d
        last = tf.matmul(last_h, self.nasr_w1)
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)
        last = tf.reshape(last, [batch_size, 1, -1])
        m = tf.nn.sigmoid(last + tf.reshape(seq, [batch_size, -1, self.out_size]) + self.nasr_b)
        coef = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(
            self.mask, [-1, 1])
        b = self.embedding[1:]
        if not self.nonhybrid:
            ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [batch_size, -1, 1]) * seq_h, 1),
                            tf.reshape(last, [-1, self.out_size])], -1)
            self.B = tf.get_variable('B', [2 * self.out_size, self.out_size],
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            y1 = tf.matmul(ma, self.B)
            logits = tf.matmul(y1, b, transpose_b=True)
        else:
            ma = tf.reduce_sum(tf.reshape(coef, [batch_size, -1, 1]) * seq_h, 1)
            logits = tf.matmul(ma, b, transpose_b=True)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar - 1, logits=logits))
        self.vars = tf.trainable_variables()
        if train:
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.L2
            loss = loss + lossL2
        return loss, logits

    def run(self, fetches, feed_dic):
        return self.sess.run(fetches, feed_dic)


class GGNN(Model):
    def __init__(self,hidden_size=100, out_size=100, batch_size=100,
                 lr=0.001, l2=0.00001, step=1, lr_dc=0.1, lr_dc_step=3, nonhybrid=True, epoch_n=30):
        super(GGNN,self).__init__(hidden_size, out_size, batch_size, nonhybrid)

        
        self.L2 = l2
        self.step = step
        self.lr_dc_step = lr_dc_step
        self.nonhybrid = nonhybrid
        self.lr_dc = lr_dc        
        self.lr = lr 
        
        self.epoch_n = epoch_n
        # updated while recommending
        self.session = -1
        self.session_items = []
        self.test_idx = 0
    
    def init_model(self, trainset_size):
        
        self.mask = tf.placeholder(dtype=tf.float32)
        self.alias = tf.placeholder(dtype=tf.int32)  # 给给每个输入重新
        self.item = tf.placeholder(dtype=tf.int32)   # 重新编号的序列构成的矩阵
        self.tar = tf.placeholder(dtype=tf.int32)

        self.nasr_w1 = tf.get_variable('nasr_w1', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.get_variable('nasrv', [1, self.out_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.get_variable('nasr_b', [self.out_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        
        
        self.embedding = tf.get_variable(shape=[self.n_nodes, self.hidden_size], name='embedding', dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.adj_in_tr = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out_tr = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_in_ts = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out_ts = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        
        self.W_in = tf.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        
        with tf.variable_scope('ggnn_model', reuse=None):
            self.loss_train, _ = self.forward(self.ggnn(self.batch_size, self.adj_in_tr, self.adj_out_tr),self.batch_size)
        with tf.variable_scope('ggnn_model', reuse=True):
            self.loss_test, self.score_test = self.forward(self.ggnn(self.batch_size, self.adj_in_ts, self.adj_out_ts),self.batch_size, train=False)
            
        self.global_step = tf.Variable(0)
        decay_steps = (self.lr_dc_step * trainset_size) / self.batch_size
        self.learning_rate = tf.train.exponential_decay(self.lr, global_step=self.global_step, decay_steps=decay_steps,
                                                        decay_rate=self.lr_dc, staircase=True)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_train, global_step=self.global_step)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
    
    def ggnn(self, batch_size, adj_in, adj_out):
        fin_state = tf.nn.embedding_lookup(self.embedding, self.item)
        cell = tf.nn.rnn_cell.GRUCell(self.out_size)
        with tf.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [batch_size, -1, self.out_size])
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                    self.W_in) + self.b_in, [batch_size, -1, self.out_size])
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out, [batch_size, -1, self.out_size])
                av = tf.concat([tf.matmul(adj_in, fin_state_in),
                                tf.matmul(adj_out, fin_state_out)], axis=-1)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2*self.out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.out_size]))
        return tf.reshape(fin_state, [batch_size, -1, self.out_size])

    def prepare_data(self, train_df, test_df, eval_sessions_neg_samples, method='ggnn'):
        self.n_nodes = len( train_df['ItemId'].unique() ) + 1
        train_data, test_data, self.item_dict, self.reversed_item_dict, count_clicks_in_test_items_not_in_train_set \
                             = prepare_data(train_df, test_df, 
                                            eval_sessions_neg_samples=eval_sessions_neg_samples)

        train_data = Data(train_data, sub_graph=True, method=method, shuffle=True)
        test_data = Data(test_data, sub_graph=True, method=method, shuffle=False, include_neg_samples=True)
        return train_data, test_data, count_clicks_in_test_items_not_in_train_set


    def fit(self, train_data):                

        with tf.Graph().as_default():

            trainset_size = len(train_data.inputs)
            self.init_model(trainset_size)
            
            for epoch in range(self.epoch_n):                
                slices = train_data.generate_batch(self.batch_size)
                #print('len(slices)', len(slices))

                fetches = [self.opt, self.loss_train, self.global_step]
                loss_ = []
                for i, j in zip(slices, np.arange(len(slices))):
                    adj_in, adj_out, alias, item, mask, targets = train_data.get_slice(i)
                    feed_dict = {self.tar: targets, self.item: item, self.adj_in_tr: adj_in,
                                 self.adj_out_tr: adj_out, self.alias: alias, self.mask: mask}
                    _, loss, _ = self.run(fetches, feed_dict)

                    loss_.append(loss)

                print('Epoch: {} - train loss: {}'.format(epoch, np.mean(loss_)))


    def evaluate(self, test_data, streaming_metrics, clicked_items_state, count_clicks_in_test_items_not_in_train_set, min_timestamp_testset):

        #for m in streaming_metrics:
        #    m.reset()
        
        with tf.Graph().as_default():

            slices = test_data.generate_batch(self.batch_size)

            test_loss_ = []
            
            all_scores = None
            for i, j in zip(slices, np.arange(len(slices))):
                
                #adj_in, adj_out, alias, item, mask, targets = self.test_data.get_slice(i)
                adj_in, adj_out, alias, item, mask, targets, negative_samples = test_data.get_slice(i)
                feed_dict = {self.tar: targets, self.item: item, self.adj_in_ts: adj_in,
                             self.adj_out_ts: adj_out, self.alias: alias, self.mask: mask}
                scores, test_loss = self.run([self.score_test, self.loss_test], feed_dict)
                
                #print('scores',scores.shape)                    

                test_loss_.append(test_loss)

                if all_scores is None:
                    all_scores = scores
                else:
                    all_scores = np.concatenate( [all_scores, scores] )



                #Sorting items by relevance (decreasing order) and summing 1 because item ids start at 1
                sorted_items_by_pred_relevance_batch = np.argsort(scores, 1)[:,::-1] + 1
                valid_items_batch = list([[label] + list(neg_samples) for label, neg_samples in zip(targets, negative_samples)])

                
                sorted_valid_items = np.array([list(filter(lambda x: x in valid_items, items)) \
                                                for items, valid_items in zip(sorted_items_by_pred_relevance_batch, valid_items_batch)])
                print('sorted_valid_items', sorted_valid_items.shape)

                #Processing ids for metrics calculation
                item_ids_to_original_vect = np.vectorize(lambda x: self.reversed_item_dict[x])
                clicked_items = np.asarray(item).flatten()
                clicked_items = clicked_items[np.nonzero(clicked_items)]
                clicked_items_original = item_ids_to_original_vect(clicked_items)
                labels_original_ids = item_ids_to_original_vect([targets])
                preds_original_ids = np.expand_dims(item_ids_to_original_vect(sorted_valid_items), 0)
                labels_norm_pop = clicked_items_state.get_articles_recent_pop_norm()[labels_original_ids]
                preds_norm_pop = clicked_items_state.get_articles_recent_pop_norm()[preds_original_ids]
                
                update_metrics(preds_original_ids, labels_original_ids, labels_norm_pop, preds_norm_pop, 
                               clicked_items_original, streaming_metrics)

                '''
                for m in streaming_metrics:
                    m.add(np.expand_dims(sorted_valid_items, 0), [targets])
                '''

                
                #Concatenating batch clicked items and labels
                #clicked_items = np.asarray(item).flatten()
                #clicked_items = clicked_items[np.nonzero(clicked_items)]
                #all_items = np.concatenate([clicked_items, targets])        
                #print('all_items', all_items.shape)

                #As all subsequences of a session are generated, taking labels as clicked items will ignore only the first clicked item, and is a good approximation of the popularity
                #clicked_items = np.array([self.reversed_item_dict[t] for t in targets])                
                #Using the lower timestamp from test set as baseline for truncating the last hour
                clicked_items_state.update_items_state(labels_original_ids[0], np.array([min_timestamp_testset+1] * labels_original_ids[0].shape[0]))


            test_loss = np.mean(test_loss_)  

           
            #If there are clicks in test set items not viewed during training, as the method is not able
            #to predict those items, assume that the recommendation was not correct in metrics
            if count_clicks_in_test_items_not_in_train_set > 0:
                perc_test_items_not_found = count_clicks_in_test_items_not_in_train_set / (len(test_data.inputs)+count_clicks_in_test_items_not_in_train_set)
                print('{} ({}%) test set clicks in items not present in train set.'.format(count_clicks_in_test_items_not_in_train_set, perc_test_items_not_found))

                #Include additional prediction errors (for accuracy metrics) when  next-clicked item is not available in train set
                for metric in streaming_metrics: 
                    if metric.name in [HitRate.name, MRR.name]:
                        fake_targets = [[1]*count_clicks_in_test_items_not_in_train_set]    
                        fake_preds = np.array([[[0]]*count_clicks_in_test_items_not_in_train_set])
                        metric.add(fake_preds, fake_targets)


            metric_results = compute_metrics_results(streaming_metrics, recommender='sr-gnn')

            print('test_loss:\t%4f\t %s'%
                  (test_loss, metric_results))
             
            
            
            
            #self.all_scores = all_scores
            #print("all_scores", self.all_scores.shape)
            
            '''
            slices = test_data.generate_batch(1)
            
            self.predicted_item_ids = []
            for idx in range(len(self.all_scores[0])):
                self.predicted_item_ids.append(int(self.reversed_item_dict[idx + 1]))  # because in item_dic, indexes start from 1 (not 0)

            print('self.predicted_item_ids', len(self.predicted_item_ids))
            '''

        return metric_results

    '''
    def fit_and_evaluate(self, train, test=None, sample_store=10000000, eval_sessions_neg_samples=None, 
            eval_top_k=10, streaming_metrics=None):

        method = 'ggnn'
        
        self.n_nodes = len( train.ItemId.unique() ) + 1

        #train_data, test_data, self.item_dict, self.reversed_item_dict = prepare_data(train, test)
        train_data, test_data, self.item_dict, self.reversed_item_dict = prepare_data(train, test, 
                                                                        eval_sessions_neg_samples=eval_sessions_neg_samples)
        
        #print('train', len(train))        
        #print('len(self.test_data.inputs)', len(test_data.inputs))
        #print('train_data', len(train_data[0]))
        #return
        
        self.train_data = Data(train_data, sub_graph=True, method=method, shuffle=True)
        self.test_data = Data(test_data, sub_graph=True, method=method, shuffle=False, include_neg_samples=True)
        print("self.test_data", len(self.test_data.inputs))
        
        
        self.decay = self.lr_dc_step * len(self.train_data.inputs) / self.batch_size


        for m in streaming_metrics:
            m.reset()
        
        with tf.Graph().as_default():
        
            self.init_model()
            
            for epoch in range(self.epoch_n):
            # print('epoch: ', epoch, '===========================================')
                slices = self.train_data.generate_batch(self.batch_size)
                #print('len(slices)', len(slices))

                fetches = [self.opt, self.loss_train, self.global_step]
                print('start training: ', datetime.datetime.now())
                loss_ = []
                for i, j in zip(slices, np.arange(len(slices))):
                    adj_in, adj_out, alias, item, mask, targets = self.train_data.get_slice(i)
                    feed_dict = {self.tar: targets, self.item: item, self.adj_in_tr: adj_in,
                                 self.adj_out_tr: adj_out, self.alias: alias, self.mask: mask}
                    _, loss, _ = self.run(fetches, feed_dict)
                
                
                print('start predicting: ', datetime.datetime.now())

                slices = self.test_data.generate_batch(self.batch_size)

                hit, mrr, test_loss_ = [], [],[]
                
                all_scores = None
                for i, j in zip(slices, np.arange(len(slices))):
                    
                    #adj_in, adj_out, alias, item, mask, targets = self.test_data.get_slice(i)
                    adj_in, adj_out, alias, item, mask, targets, negative_samples = self.test_data.get_slice(i)
                    feed_dict = {self.tar: targets, self.item: item, self.adj_in_ts: adj_in,
                                 self.adj_out_ts: adj_out, self.alias: alias, self.mask: mask}
                    scores, test_loss = self.run([self.score_test, self.loss_test], feed_dict)
                    
                    #print('scores',scores.shape)                    

                    test_loss_.append(test_loss)
                    if all_scores is None:
                        all_scores = scores
                    else:
                        all_scores = np.concatenate( [all_scores, scores] )


                    #print("targets", len(targets))
                    #print("negative_samples", len(negative_samples))


                    #Sorting items by relevance (decreasing order) and summing 1 because item ids start at 1
                    sorted_items_by_pred_relevance_batch = np.argsort(scores, 1)[:,::-1] + 1
                    valid_items_batch = list([[label] + list(neg_samples) for label, neg_samples in zip(targets, negative_samples)])

                    #print("targets[:2]", targets[:2])
                    #print("negative_samples[:2]", negative_samples[:2])
                    #print("valid_items_batch[:2]", valid_items_batch[:2])

                    
                    #print('valid_items.lenth', len(valid_items_batch))
                    #print('valid_items', valid_items_batch[:5])
                    top_sorted_valid_items = np.array([list(filter(lambda x: x in valid_items, items))[:eval_top_k] \
                                                    for items, valid_items in zip(sorted_items_by_pred_relevance_batch, valid_items_batch)])
                    
                    #print('top_sorted_valid_items', top_sorted_valid_items)

                    #for idx1, line in enumerate(top_sorted_valid_items[:3]):
                    #    for idx2, item in enumerate(line):
                    #        print(scores[idx1, item-1])
                    #    print()
                    

                    #print('sorted_valid_items.length', len(sorted_valid_items))
                    #print('sorted_valid_items', sorted_valid_items[:5])

                    #print('targets', targets[:5])
                    #print('top_items', top_sorted_valid_items[:5])

                    for m in streaming_metrics:
                        m.add(np.expand_dims(top_sorted_valid_items, 0), [targets])

                    

                
                #hit = np.mean(hit)
                #mrr = np.mean(mrr)
                #test_loss = np.mean(test_loss_)
                #print('train_loss:\t%.4f\ttest_loss:\t%4f\tRecall@%d:\t%.4f\tMMR@%d:\t%.4f\tEpoch:\t%d'%
                #      (loss, test_loss, eval_top_k, hit, eval_top_k, mrr, epoch))


                test_loss = np.mean(test_loss_)

                metric_results = {}
                for m in streaming_metrics:
                    metric_results[m.name] = m.result()

                print('Epoch:\t%d\t train_loss:\t%.4f\t test_loss:\t%4f\t %s'%
                      (epoch, loss, test_loss, metric_results))
                
                
                self.all_scores = all_scores
                print("all_scores", self.all_scores.shape)
            
            
            #slices = self.test_data.generate_batch(1)
            
            #self.predicted_item_ids = []
            #for idx in range(len(self.all_scores[0])):
            #    self.predicted_item_ids.append(int(self.reversed_item_dict[idx + 1]))  # because in item_dic, indexes start from 1 (not 0)

            #print('self.predicted_item_ids', len(self.predicted_item_ids))
            

            return metric_results

    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, type='view', timestamp=0):

        if (self.session != session_id):  # new session
            self.session = session_id
            self.session_items = list()
# 
#         # convert original item_id according to the item_dic
        item_id_dic = self.item_dict[input_item_id]
        self.session_items.append(item_id_dic)
#         adj_in, adj_out, alias, item, mask, targets = self.test_data.get_slice_by_session_items(self.session_items, self.test_data.len_max)
#         feed_dict = {self.tar: targets, self.item: item, self.adj_in_ts: adj_in,
#                      self.adj_out_ts: adj_out, self.alias: alias, self.mask: mask}
#         scores, test_loss = self.run([self.score_test, self.loss_test], feed_dict)
        # index_ascending_order = np.argsort(scores) # default: axis= 1 , [:, -20:]
        # index = np.flip(index_ascending_order)
        # preds_ascending_order = np.sort(scores) # default: axis= 1 , [:, -20:]
        # preds = np.flip(preds_ascending_order)
        # predict_for_item_ids = []
        # # retrieve original item_id according to the item_dic
        # for idx in index[0]:
        #     predict_for_item_ids += [self.reversed_item_dict[idx+1]]   # because in item_dic, indexes start from 1 (not 0)
        # series = pd.Series(data=preds[0], index=predict_for_item_ids)
        
#         #test that the scores are correct
#         adj_in, adj_out, alias, item, mask, targets = self.test_data.get_slice([self.test_idx])
#         print( 'predict_next' )
#         print( self.session_items )
#         print( item )
        
        scores = self.all_scores[self.test_idx]
        self.test_idx += 1
        
#         predicted_item_ids = []
#         for idx in range(len(scores)):
#             predicted_item_ids.append(int(self.reversed_item_dict[idx + 1]))  # because in item_dic, indexes start from 1 (not 0)
            
        series = pd.Series(data=scores, index=self.predicted_item_ids)
        return series
    '''
