import numpy as np

import tensorflow as tf


class ACR_Model:
    
    def __init__(self, text_feature_extractor, features, metadata_features, metadata_feature_columns, labels, mode, params):
        self.params = params
        dropout_keep_prob = params['dropout_keep_prob']
        l2_reg_lambda = params['l2_reg_lambda']

        training = mode == tf.estimator.ModeKeys.TRAIN

        with tf.variable_scope("main", initializer=tf.contrib.layers.variance_scaling_initializer()):
            with tf.variable_scope("input_article_metadata"):
                metadata_features = tf.feature_column.input_layer(metadata_features, 
                                                                  metadata_feature_columns)
                print("Metadata features: {}".format(metadata_features.get_shape()))


            with tf.variable_scope("input_word_embeddings"):
                input_text_layer = tf.contrib.layers.embed_sequence(
                                ids=features['text'], 
                                vocab_size=params['vocab_size'],
                                embed_dim=params['word_embedding_size'],
                                trainable=False,        
                                initializer=params['embedding_initializer']
                                )

            with tf.variable_scope("textual_features_representation"):
                if text_feature_extractor.upper() == 'CNN':
                    content_features = self.cnn_feature_extractor(input_text_layer)
                elif text_feature_extractor.upper() == 'RNN':
                    content_features = self.rnn_feature_extractor(input_text_layer, features['text_length'])
                else:
                    raise Exception('Text feature extractor option invalid! Valid values are: CNN, RNN')

                content_metadata_features = tf.concat([content_features, metadata_features], axis=-1)
                #print('content_metadata_features.shape', content_metadata_features.get_shape())
                
                dropout_conv_layers_concat = tf.layers.dropout(inputs=content_metadata_features, 
                                                   rate=1.0-dropout_keep_prob, 
                                                   training=training)

                with tf.variable_scope("article_content_embedding"):
                    hidden = tf.layers.dense(inputs=dropout_conv_layers_concat, units=params['acr_embeddings_size'], 
                        activation=tf.nn.tanh,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda),
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
                    self.article_content_embedding = hidden
                    #print('article_content_embedding.shape', self.article_content_embedding.get_shape())

            
            with tf.variable_scope("metadata_prediction"):
                dropout_hidden = tf.layers.dropout(inputs=hidden, 
                                                   rate=1.0-dropout_keep_prob, 
                                                   training=training)

                logits = tf.layers.dense(inputs=dropout_hidden, units=params['classes_count'],
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda))
                #print('logits.shape', logits.get_shape())
                
                self.predictions = tf.argmax(logits, 1)
            
            
            if mode != tf.estimator.ModeKeys.PREDICT:
                with tf.variable_scope("loss"):
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
                    tf.summary.scalar('cross_entropy_loss', family='train', tensor=loss)

                    reg_loss = tf.losses.get_regularization_loss()
                    tf.summary.scalar("reg_loss", family='train', tensor=reg_loss)

                    self.total_loss = loss  + reg_loss
                    tf.summary.scalar("total_loss", family='train', tensor=self.total_loss)
                
                if mode == tf.estimator.ModeKeys.TRAIN:
                    with tf.variable_scope("training"):
                        self.train_op = tf.contrib.layers.optimize_loss(
                                                            loss=self.total_loss,
                                                            optimizer="Adam",
                                                            learning_rate=params['learning_rate'],
                                                            #learning_rate_decay_fn=lambda lr, gs: tf.train.exponential_decay(params['learning_rate'], tf.train.get_global_step(), 100, 0.96, staircase=True),
                                                            global_step=tf.train.get_global_step(),
                                                            summaries=["learning_rate", "global_gradient_norm"])

                with tf.variable_scope("eval_metrics"):
                    self.accuracy, self.accuracy_update_op = \
                            tf.metrics.accuracy(predictions=self.predictions, 
                                                labels=labels,
                                                name='accuracy')           


                with tf.variable_scope("stats"):
                    nops = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
                    tf.logging.info('Number of trainable parameters {}'.format(nops))

    def cnn_feature_extractor(self, input_text_layer):
        with tf.variable_scope("CNN"):
            conv_layers = []
            for kernel_sizes in map(int, self.params['cnn_filter_sizes'].split(',')):
                conv = tf.layers.conv1d(
                                    inputs=input_text_layer,
                                    filters=self.params['cnn_num_filters'],
                                    kernel_size=kernel_sizes,
                                    padding="valid",
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.params['l2_reg_lambda']))
                # Max Pooling over time (words)
                pool = tf.reduce_max(input_tensor=conv, axis=1)

                conv_layers.append(pool)              
            
            conv_layers_concat = tf.concat(conv_layers, axis=-1)
            return conv_layers_concat


    def rnn_feature_extractor(self, input_text_layer, text_lengths):
        with tf.variable_scope("LSTM"):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(384)

            # create the complete LSTM
            rnn_outputs, final_states = tf.nn.dynamic_rnn(
                lstm_cell, input_text_layer, sequence_length=text_lengths, dtype=tf.float32)

            # get the final hidden states of dimensionality [batch_size x rnn_units]
            rnn_final_states = final_states.h
            return rnn_final_states
