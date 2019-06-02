import numpy as np

import tensorflow as tf


def multi_label_predictions_binarizer(predictions, threshold=0.5):
    predictions = tf.sigmoid(predictions)
    return tf.cast(tf.greater(predictions, threshold), tf.int64)

class ACR_Model:
    
    def __init__(self, text_feature_extractor, features, 
                 metadata_input_features, metadata_input_feature_columns, labels, labels_features_config, mode, params):
        self.params = params
        dropout_keep_prob = params['dropout_keep_prob']
        l2_reg_lambda = params['l2_reg_lambda']

        training = mode == tf.estimator.ModeKeys.TRAIN

        with tf.variable_scope("main", initializer=tf.contrib.layers.variance_scaling_initializer()):
            input_features = []

            #Creating a tensor for class weights of each label
            labels_classes_weights = {}
            if 'labels_class_weights' in params:
                for label_column_name in params['labels_class_weights']:
                    labels_classes_weights[label_column_name] = tf.constant(params['labels_class_weights'][label_column_name], tf.float32)

            with tf.variable_scope("input_article_metadata"):
                #If there is articles metadata available for the model
                if metadata_input_features is not None and len(metadata_input_features) > 0:
                    metadata_features = tf.feature_column.input_layer(metadata_input_features, 
                                                                    metadata_input_feature_columns)

                    tf.logging.info("Metadata features shape: {}".format(metadata_features.get_shape()))

                    #Creating a FC layer on top of metadata features (usually OHE categorical features) to make it dense (like the CNN output)
                    num_metadata_inputs = max(int(int(metadata_features.get_shape()[-1]) / 4), 2)
                    hidden_metadata = tf.layers.dense(inputs=metadata_features, units=num_metadata_inputs, 
                                             activation=tf.nn.relu,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda))

                    tf.logging.info("Hidden Metadata features shape: {}".format(hidden_metadata.get_shape()))

                    input_features.append(hidden_metadata)
                    

                    #tf.summary.histogram('multi_hot', 
                    #              values=tf.reduce_sum(metadata_features[:,1:], axis=1))


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

                input_features.append(content_features)

                input_features_concat = tf.concat(input_features, axis=-1)
                tf.logging.info("input_features_concat.shape={}".format(input_features_concat.get_shape()))

                dropout_conv_layers_concat = tf.layers.dropout(inputs=input_features_concat,#input_features_concat, 
                                                   rate=1.0-dropout_keep_prob, 
                                                   training=training)

                #Testing a new FC
                fc2 = tf.layers.dense(inputs=dropout_conv_layers_concat, units=params['acr_embeddings_size'], 
                                             activation=tf.nn.relu,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda))

                with tf.variable_scope("article_content_embedding"):
                    hidden = tf.layers.dense(inputs=fc2, units=params['acr_embeddings_size'], 
                                             activation=tf.nn.tanh,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda),
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
                    self.article_content_embedding = hidden
                    #print('article_content_embedding.shape', self.article_content_embedding.get_shape())

            
            with tf.variable_scope("network_output"):
                dropout_hidden = tf.layers.dropout(inputs=hidden, 
                                                   rate=1.0-dropout_keep_prob, 
                                                   training=training)

                labels_logits = {}
                self.labels_predictions = {}

                for label_feature_name in labels_features_config:
                    with tf.variable_scope("output_{}".format(label_feature_name)):

                        label_feature = labels_features_config[label_feature_name]

                        labels_logits[label_feature_name] = tf.layers.dense(inputs=dropout_hidden, units=label_feature['cardinality'],
                                                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_lambda),
                                                                            activation=None)

                        if labels_features_config[label_feature_name]['classification_type'] == 'multiclass':
                            self.labels_predictions[label_feature_name] = tf.argmax(labels_logits[label_feature_name], 1)
                        
                        elif labels_features_config[label_feature_name]['classification_type'] == 'multilabel':
                            #If its a multi-label head, convert labels in multi-hot representation
                            self.labels_predictions[label_feature_name] = multi_label_predictions_binarizer(labels_logits[label_feature_name])

            if mode != tf.estimator.ModeKeys.PREDICT:
                with tf.variable_scope("loss"):
                    loss = tf.constant(0.0, tf.float32)

                    for label_feature_name in labels_features_config:
                        if labels_features_config[label_feature_name]['classification_type'] == 'multiclass':
                            #label_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=labels_logits[label_feature_name], 
                            #                                                                        labels=labels[label_feature_name]))
                            
                            #If the label feature have classes weights, use the weights to deal with unbalanced classes
                            weights=tf.constant(1.0)
                            if label_feature_name in labels_classes_weights:
                                weights=tf.gather(labels_classes_weights[label_feature_name], labels[label_feature_name])
                                
                            label_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=labels_logits[label_feature_name], 
                                                                                           labels=labels[label_feature_name],
                                                                                           weights=weights))
                            tf.summary.scalar('softmax_cross_entropy_loss-{}'.format(label_feature_name), 
                                          family='train', tensor=label_loss)

                        elif labels_features_config[label_feature_name]['classification_type'] == 'multilabel':
                            #Tuning the label into multi-hot representation
                            labels[label_feature_name] = tf.reduce_sum(tf.one_hot(indices=labels[label_feature_name], 
                                                                       depth=labels_features_config[label_feature_name]['cardinality']
                                                                       ), reduction_indices=1)
                            
                            #Forcing that label of padding value is always zero
                            labels[label_feature_name] = tf.multiply(labels[label_feature_name],
                                                                     tf.concat([tf.constant([0], tf.float32), tf.ones(tf.shape(labels[label_feature_name])[-1]-1, tf.float32)], axis=0))

                            #tf.summary.histogram('multilabel_count_{}'.format(label_feature_name), 
                            #      values=tf.reduce_sum(labels[label_feature_name], axis=1))

                            tf.logging.info("labels_multi_hot.shape = {}".format(labels[label_feature_name].get_shape()))

                            label_loss = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[label_feature_name],
                                                                    logits=labels_logits[label_feature_name]), axis=1))
                            tf.summary.scalar('sigmoid-cross_entropy_loss-{}'.format(label_feature_name), 
                                          family='train', tensor=label_loss)

                        feature_weight_on_loss = labels_features_config['feature_weight_on_loss'] if 'feature_weight_on_loss' in labels_features_config else 1.0
                        loss += feature_weight_on_loss * label_loss                    
                    
                    tf.summary.scalar('cross_entropy_loss-total', family='train', tensor=loss)

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

                    self.eval_metrics = {}

                    for label_feature_name in labels_features_config:

                        #tf.logging.info("{}-labels_predictions: {}".format(label_feature_name, self.labels_predictions[label_feature_name].get_shape()))
                        #tf.logging.info("{}-labels: {}".format(label_feature_name, labels[label_feature_name].get_shape()))

                        if labels_features_config[label_feature_name]['classification_type'] == 'multiclass':
                            accuracy, accuracy_update_op = \
                                    tf.metrics.accuracy(predictions=self.labels_predictions[label_feature_name], 
                                                        labels=labels[label_feature_name],
                                                        name="accuracy-{}".format(label_feature_name))  

                            self.eval_metrics["accuracy-{}".format(label_feature_name)] = (accuracy, accuracy_update_op)

                        elif labels_features_config[label_feature_name]['classification_type'] == 'multilabel':

                            precision, precision_update_op = \
                                    tf.metrics.precision(predictions=self.labels_predictions[label_feature_name], 
                                                        labels=labels[label_feature_name],
                                                        name="precision-{}".format(label_feature_name))  

                            self.eval_metrics["precision-{}".format(label_feature_name)] = (precision, precision_update_op)

                            recall, recall_update_op = \
                                    tf.metrics.recall(predictions=self.labels_predictions[label_feature_name], 
                                                        labels=labels[label_feature_name],
                                                        name="recall-{}".format(label_feature_name))  

                            self.eval_metrics["recall-{}".format(label_feature_name)] = (recall, recall_update_op)



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
