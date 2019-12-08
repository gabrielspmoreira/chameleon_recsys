import numpy as np

import tensorflow as tf


def multi_label_predictions_binarizer(predictions, threshold=0.5):
    predictions = tf.sigmoid(predictions)
    return tf.cast(tf.greater(predictions, threshold), tf.int64)


def state_tuples_to_cudnn_lstm_state(lstm_state_tuples):
    """Convert LSTMStateTuples to CudnnLSTM format."""
    c = tf.stack([s.c for s in lstm_state_tuples])
    h = tf.stack([s.h for s in lstm_state_tuples])
    return (c, h)


def cudnn_lstm_state_to_state_tuples(cudnn_lstm_state):
    """Convert CudnnLSTM format to LSTMStateTuples."""
    c, h = cudnn_lstm_state
    return tuple(
        tf.contrib.rnn.LSTMStateTuple(h=h_i, c=c_i)
        for c_i, h_i in zip(tf.unstack(c), tf.unstack(h)))    


def cosine_sim_v2(x1, x2,name = 'Cosine_loss'):
    with tf.name_scope(name):
        x1_norm = tf.nn.l2_normalize(x1, axis=-1)
        x2_norm = tf.nn.l2_normalize(x2, axis=-1)
        num = tf.matmul(x1_norm,x2_norm, transpose_b=True)
        return num

class ACR_Model:
    
    def __init__(self, training_task, text_feature_extractor, features, 
                 metadata_input_features, metadata_input_feature_columns, labels, labels_features_config, mode, params):
        self.params = params
        self.training = mode == tf.estimator.ModeKeys.TRAIN
        self.mode = mode
        self.text_feature_extractor = text_feature_extractor.upper()
        self.labels_features_config = labels_features_config
        self.acr_embeddings_size = params['acr_embeddings_size']
        self.dropout_keep_prob = params['dropout_keep_prob']
        self.l2_reg_lambda = params['l2_reg_lambda']       
        self.rnn_units = params['rnn_units']
        self.rnn_layers = params['rnn_layers']
        self.rnn_direction = params['rnn_direction']
        self.special_token_embedding_vector = params['special_token_embedding_vector']
        self.autoencoder_noise = params['autoencoder_noise']
        self.features = features
        self.labels = labels 


        self.article_id = features['article_id']


        with tf.variable_scope("main", initializer=tf.contrib.layers.xavier_initializer()):
            '''
            input_features = []            
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
            '''

            with tf.variable_scope("input_word_embeddings"):                                
                self.word_embeddings_matrix = tf.constant(self.params['word_embeddings_matrix'],                                                     
                                                            dtype=tf.float32)

                input_text_embeddings = tf.nn.embedding_lookup(self.word_embeddings_matrix, features['text'])
                tf.logging.info('input_text_layer shape: {}'.format(input_text_embeddings.get_shape()))


            if training_task.lower() ==  'metadata_classification':
                self.build_graph_metadata_classification(input_text_embeddings)


            elif training_task.lower() ==  'autoencoder':
                self.build_graph_autoencoder(input_text_embeddings)

            else:
                raise Exception('Training task can be only: (metadata_classification | autoencoder)')
                


    def build_graph_metadata_classification(self, input_text_embeddings):
        labels_features_config = self.labels_features_config
        text_feature_extractor = self.text_feature_extractor


        with tf.variable_scope("metadata_classification"):

            with tf.variable_scope("textual_feature_extraction"):
                if text_feature_extractor == 'CNN':
                    content_features = self.cnn_feature_extractor(input_text_embeddings)

                elif text_feature_extractor in ['LSTM', 'GRU']:
                    #Reversing input feature for better performance (because usually most relevant words are in the start of the document)
                    input_text_embeddings_reversed = tf.reverse(input_text_embeddings, axis=[1])

                    rnn_outputs, _, _ = self.build_cudnn_rnn(rnn_type=text_feature_extractor, 
                                                input_text_embeddings=input_text_embeddings_reversed, 
                                                text_lengths=self.features['text_length'],
                                                suffix='rnn')

                    #Max pool on time (words) dimension of the output
                    content_features = tf.reduce_max(rnn_outputs, axis=1)

                else:
                    raise Exception('Text feature extractor option invalid! Valid values are: CNN, LSTM, GRU')

                
                #input_features.append(content_features)
                #input_features_concat = tf.concat(input_features, axis=-1)

                input_features_concat = content_features

                tf.logging.info("input_features_concat.shape={}".format(input_features_concat.get_shape()))

                dropout_conv_layers_concat = tf.layers.dropout(inputs=input_features_concat,
                                                   rate=1.0-self.dropout_keep_prob, 
                                                   training=self.training)

                fc2 = tf.layers.dense(inputs=dropout_conv_layers_concat, units=self.acr_embeddings_size, 
                                             activation=tf.nn.relu,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

                with tf.variable_scope("article_content_embedding"):
                    hidden = tf.layers.dense(inputs=fc2, units=self.acr_embeddings_size, 
                                             activation=tf.nn.tanh,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
                    self.article_content_embedding = hidden

            
            with tf.variable_scope("network_output"):
                dropout_hidden = tf.layers.dropout(inputs=hidden, 
                                                   rate=1.0-self.dropout_keep_prob, 
                                                   training=self.training)

                labels_logits = {}
                self.labels_predictions = {}

                for label_feature_name in labels_features_config:
                    with tf.variable_scope("output_{}".format(label_feature_name)):

                        label_feature = labels_features_config[label_feature_name]

                        labels_logits[label_feature_name] = tf.layers.dense(inputs=dropout_hidden, units=label_feature['cardinality'],
                                                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
                                                                            activation=None)

                        if labels_features_config[label_feature_name]['classification_type'] == 'multiclass':
                            self.labels_predictions[label_feature_name] = tf.argmax(labels_logits[label_feature_name], 1)
                        
                        elif labels_features_config[label_feature_name]['classification_type'] == 'multilabel':
                            #If its a multi-label head, convert labels in multi-hot representation
                            self.labels_predictions[label_feature_name] = multi_label_predictions_binarizer(labels_logits[label_feature_name])

            if self.mode != tf.estimator.ModeKeys.PREDICT:

                with tf.variable_scope("loss"):
                    loss = tf.constant(0.0, tf.float32)

                    #Creating a tensor for class weights of each label
                    labels_classes_weights = {}
                    if 'labels_class_weights' in self.params:
                        for label_column_name in self.params['labels_class_weights']:
                            labels_classes_weights[label_column_name] = tf.constant(self.params['labels_class_weights'][label_column_name], tf.float32)        


                    for label_feature_name in labels_features_config:
                        if labels_features_config[label_feature_name]['classification_type'] == 'multiclass':
                            #If the label feature have classes weights, use the weights to deal with unbalanced classes
                            weights=tf.constant(1.0)
                            if label_feature_name in labels_classes_weights:
                                weights=tf.gather(labels_classes_weights[label_feature_name], self.labels[label_feature_name])
                                
                            label_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=labels_logits[label_feature_name], 
                                                                                           labels=self.labels[label_feature_name],
                                                                                           weights=weights))
                            tf.summary.scalar('softmax_cross_entropy_loss-{}'.format(label_feature_name), 
                                          family='train', tensor=label_loss)

                        elif labels_features_config[label_feature_name]['classification_type'] == 'multilabel':
                            #Tuning the label into multi-hot representation
                            self.labels[label_feature_name] = tf.reduce_sum(tf.one_hot(indices=self.labels[label_feature_name], 
                                                                       depth=labels_features_config[label_feature_name]['cardinality']
                                                                       ), reduction_indices=1)
                            
                            #Forcing that label of padding value is always zero
                            self.labels[label_feature_name] = tf.multiply(self.labels[label_feature_name],
                                                                     tf.concat([tf.constant([0], tf.float32), tf.ones(tf.shape(self.labels[label_feature_name])[-1]-1, tf.float32)], axis=0))

                            tf.logging.info("labels_multi_hot.shape = {}".format(self.labels[label_feature_name].get_shape()))

                            label_loss = tf.reduce_mean(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels[label_feature_name],
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
                
                if self.mode == tf.estimator.ModeKeys.TRAIN:
                    with tf.variable_scope("training"):
                        self.train_op = tf.contrib.layers.optimize_loss(
                                                            loss=self.total_loss,
                                                            optimizer="Adam",
                                                            learning_rate=self.params['learning_rate'],
                                                            #learning_rate_decay_fn=lambda lr, gs: tf.train.exponential_decay(params['learning_rate'], tf.train.get_global_step(), 100, 0.96, staircase=True),
                                                            global_step=tf.train.get_global_step(),
                                                            summaries=["learning_rate", "global_gradient_norm"])

                with tf.variable_scope("eval_metrics"):

                    self.eval_metrics = {}

                    for label_feature_name in labels_features_config:

                        if labels_features_config[label_feature_name]['classification_type'] == 'multiclass':
                            accuracy, accuracy_update_op = \
                                    tf.metrics.accuracy(predictions=self.labels_predictions[label_feature_name], 
                                                        labels=self.labels[label_feature_name],
                                                        name="accuracy-{}".format(label_feature_name))  

                            self.eval_metrics["accuracy-{}".format(label_feature_name)] = (accuracy, accuracy_update_op)

                        elif labels_features_config[label_feature_name]['classification_type'] == 'multilabel':

                            precision, precision_update_op = \
                                    tf.metrics.precision(predictions=self.labels_predictions[label_feature_name], 
                                                        labels=self.labels[label_feature_name],
                                                        name="precision-{}".format(label_feature_name))  

                            self.eval_metrics["precision-{}".format(label_feature_name)] = (precision, precision_update_op)

                            recall, recall_update_op = \
                                    tf.metrics.recall(predictions=self.labels_predictions[label_feature_name], 
                                                        labels=self.labels[label_feature_name],
                                                        name="recall-{}".format(label_feature_name))  

                            self.eval_metrics["recall-{}".format(label_feature_name)] = (recall, recall_update_op)


                
    def cnn_feature_extractor(self, input_text_embeddings):
        with tf.variable_scope("CNN"):
            conv_layers = []
            for kernel_sizes in map(int, self.params['cnn_filter_sizes'].split(',')):
                conv = tf.layers.conv1d(
                                    inputs=input_text_embeddings,
                                    filters=self.params['cnn_num_filters'],
                                    kernel_size=kernel_sizes,
                                    padding="valid",
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
                # Max Pooling over time (words)
                pool = tf.reduce_max(input_tensor=conv, axis=1)

                conv_layers.append(pool)              
            
            conv_layers_concat = tf.concat(conv_layers, axis=-1)
            return conv_layers_concat



    def build_cudnn_rnn(self, rnn_type, input_text_embeddings, text_lengths, return_max_pool_over_outputs=True,
                              cudnn_initial_state = None, suffix=''):
        #Based on https://github.com/tensorflow/magenta/blob/master/magenta/models/shared/events_rnn_graph.py
        # and on https://github.com/mutux/ptb_lm/blob/master/lm.py
        with tf.variable_scope("RNN_{}".format(suffix)):
            batch_size=tf.shape(input_text_embeddings)[0]

            #Converting inputs from Batch-major to Time-major (requirement for CUDRNN)
            cudnn_inputs = tf.transpose(input_text_embeddings, [1, 0, 2])
            tf.logging.info('cudnn_inputs={}'.format(cudnn_inputs.get_shape()))
           

            #Keep all this Ops on GPU, to avoid error when saving CudnnGRU object in the checkpoint
            with tf.device('/gpu:0'):

                if rnn_type == 'LSTM':        
                    with tf.variable_scope("LSTM"):        
                    
                        #If do not pass initial state, initializes with zeros
                        if cudnn_initial_state == None:
                            initial_state = tuple(tf.contrib.rnn.LSTMStateTuple(
                                                    h=tf.zeros([batch_size, self.rnn_units], dtype=tf.float32),
                                                    c=tf.zeros([batch_size, self.rnn_units], dtype=tf.float32))
                                                for _ in range(self.rnn_layers * (2 if self.rnn_direction == 'bidirectional' else 1)))
                            cudnn_initial_state = state_tuples_to_cudnn_lstm_state(initial_state)     

                        
                        tf.logging.info("cudnn_initial_state={}".format(cudnn_initial_state)) 


                        cell = tf.contrib.cudnn_rnn.CudnnLSTM(
                                num_layers=self.rnn_layers,
                                num_units=self.rnn_units,
                                direction=self.rnn_direction,
                                dropout=1.0-self.dropout_keep_prob, #Dropout is applied between each layer (no dropout is applied for a model with a single layer
                                dtype=tf.float32,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                bias_initializer=tf.zeros_initializer(),
                            )

                        rnn_outputs, cudnn_final_state = cell(
                                cudnn_inputs,
                                initial_state=cudnn_initial_state,
                                training=self.training
                            )

                        #Converting back from Time-major (requirement for CUDRNN) to Batch-major 
                        rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])

                        
                        #Return hidden state from last RNN layer
                        final_state = cudnn_lstm_state_to_state_tuples(cudnn_final_state)                    

                        #Takes the last hidden state from last RNN layer
                        if self.rnn_direction == 'unidirectional':
                            last_layer_state = final_state[-1].h
                        elif self.rnn_direction == 'bidirectional':
                            #Combines last forward and backward layers
                            last_layer_state = tf.concat([final_state[-1].h,
                                                          final_state[-2].h], axis=-1)
                    

                elif rnn_type == 'GRU':
                    with tf.variable_scope("GRU"):     

                        #If do not pass initial state, initializes with zeros
                        if cudnn_initial_state == None:
                            cudnn_initial_state = (tf.zeros([self.rnn_layers * (2 if self.rnn_direction == 'bidirectional' else 1), 
                                                          batch_size, 
                                                          self.rnn_units], 
                                                         dtype=tf.float32)
                                                  , 
                                                  )

                        #GRU
                        cell = tf.contrib.cudnn_rnn.CudnnGRU(
                                    num_layers=self.rnn_layers,
                                    num_units=self.rnn_units,
                                    direction=self.rnn_direction,
                                    dropout=1.0-self.dropout_keep_prob, #Dropout is applied between each layer (no dropout is applied for a model with a single layer
                                    dtype=tf.float32,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(),
                                )                

                        rnn_outputs, cudnn_final_state = cell(
                                cudnn_inputs,
                                initial_state=cudnn_initial_state,
                                training=self.training
                            )


                        #Converting back from Time-major (requirement for CUDRNN) to Batch-major 
                        rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])

                        #Takes the last hidden state from last RNN layer
                        if self.rnn_direction == 'unidirectional':
                            last_layer_state = cudnn_final_state[0][-1]
                        elif self.rnn_direction == 'bidirectional':
                            #Combines last forward and backward layers
                            last_layer_state = tf.concat([cudnn_final_state[0][-1],
                                                          cudnn_final_state[0][-2]], axis=-1)


            #Investigate: Unlike dynamic_rnn, CudnnGRU doesn't allow you to specify sequence lengths. Still, it is over an order of magnitude faster, but you will have to be careful on how you extract your outputs (e.g. if you're interested in the final hidden state of each sequence that is padded and of varying length, you will need each sequence's length).

            return rnn_outputs, cudnn_final_state, last_layer_state



    def build_graph_autoencoder(self, input_text_embeddings):
        '''
        Based on https://github.com/erickrf/autoencoder
        '''
        text_feature_extractor = self.text_feature_extractor

        with tf.variable_scope("autoencoder"):

            with tf.variable_scope("textual_feature_extraction"):

                input_text_embeddings_transformed = input_text_embeddings
                if self.autoencoder_noise > 0:
                    input_text_embeddings_transformed = input_text_embeddings + \
                                                   tf.random_normal(shape=tf.shape(input_text_embeddings), 
                                                                    mean=0.0, stddev=self.autoencoder_noise)

                

                
                special_token_embedding = tf.constant(self.special_token_embedding_vector, dtype=tf.float32)
                #Reversing input feature for better performance (because usually most relevant words are in the start of the document)
                input_text_embeddings_reversed = tf.reverse(input_text_embeddings_transformed, axis=[1])   


    
                rnn_outputs_encoder, cudnn_final_state_encoder, last_layer_state_encoder = self.build_cudnn_rnn(
                                        rnn_type=text_feature_extractor, 
                                        input_text_embeddings=input_text_embeddings_reversed, 
                                        text_lengths=self.features['text_length'], suffix='encoder')


                final_state_compressed = tf.layers.dense(cudnn_final_state_encoder[0], self.params['acr_embeddings_size'],                                        
                                            activation=tf.nn.tanh, 
                                            #kernel_initializer=variance_scaling_initializer(),
                                            #kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay),
                                           )
                tf.logging.info('final_state_compressed.shape={}'.format(final_state_compressed.get_shape()))

                final_state_reconstructed = tf.layers.dense(final_state_compressed, self.params['rnn_units'],                                        
                                            activation=tf.nn.tanh, 
                                            #kernel_initializer=variance_scaling_initializer(),
                                            #kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay),
                                           )
                tf.logging.info('final_state_reconstructed.shape={}'.format(final_state_reconstructed.get_shape()))

                cudnn_final_state_reconstructed = (final_state_reconstructed,)

                
                #self.article_content_embedding = last_layer_state_encoder
                self.article_content_embedding = final_state_compressed[-1]





                #TODO: Prototype LSTM Autoencoder for reconstruction and prediction
                special_token_embedding_tiled = tf.tile(tf.expand_dims(special_token_embedding, 1), [tf.shape(input_text_embeddings)[0], 1, 1])
                input_decoder = tf.concat([special_token_embedding_tiled, input_text_embeddings[:,:-1,:]], axis=1)

                rnn_outputs_decoder, cudnn_final_state_decoder, last_layer_state_decoder = self.build_cudnn_rnn(rnn_type=text_feature_extractor, 
                                            input_text_embeddings=input_decoder, 
                                            text_lengths=self.features['text_length'], suffix='decoder',
                                            cudnn_initial_state=cudnn_final_state_reconstructed
                                            )


                decoder_outputs_reconstructed = tf.layers.dense(rnn_outputs_decoder, self.params['word_embedding_size'],                                        
                                            activation=None, 
                                            #kernel_initializer=variance_scaling_initializer(),
                                            #kernel_regularizer=tf.contrib.layers.l2_regularizer(self.reg_weight_decay),
                                           )


                DEBUG_FIRST_N_WORDS = 10


                #The matmul-based cosine similarity is much more efficient
                decoder_outputs_reconstructed_first_n_words = decoder_outputs_reconstructed[:,:DEBUG_FIRST_N_WORDS]
                word_embeddings_matrix_expanded = tf.tile(tf.expand_dims(self.word_embeddings_matrix, 0), [tf.shape(decoder_outputs_reconstructed_first_n_words)[0],1,1])
                cosine_sim_matrix = cosine_sim_v2(decoder_outputs_reconstructed_first_n_words, 
                                        word_embeddings_matrix_expanded)
                

                sorted_word_sims, sorted_word_ids = tf.nn.top_k(cosine_sim_matrix, k=5)
                tf.logging.info('sorted_word_ids={}'.format(sorted_word_ids.get_shape()))
                self.predicted_word_ids = sorted_word_ids



                if self.mode != tf.estimator.ModeKeys.PREDICT:         

                    #tf.logging.info('decoder_outputs_reconstructed: {}'.format(decoder_outputs_reconstructed))

                    
                    #mask = tf.cast(tf.sequence_mask(features['text_length']), tf.float32)
                    mask = tf.tile(tf.expand_dims(tf.cast(tf.sign(self.features['text']), tf.float32), -1), [1, 1, tf.shape(input_text_embeddings)[2]])

                    tf.summary.scalar('avg_valid_words', family='train', tensor=tf.reduce_mean(tf.reduce_sum(mask, axis=1)))                    
                    
                    #Computes the MSE considering the mask
                    autoencoder_reconstruction_loss_masked = tf.reduce_sum(tf.square((input_text_embeddings * mask) - (decoder_outputs_reconstructed  * mask))) / tf.reduce_sum(mask)
                    tf.summary.scalar('autoencoder_reconstruction_loss_masked', family='train', tensor=autoencoder_reconstruction_loss_masked)

                    reg_loss = tf.losses.get_regularization_loss()
                    tf.summary.scalar("reg_loss", family='train', tensor=reg_loss)

                    self.total_loss = autoencoder_reconstruction_loss_masked + reg_loss

                    if self.mode == tf.estimator.ModeKeys.TRAIN:
                        with tf.variable_scope("training"):
                                self.train_op = tf.contrib.layers.optimize_loss(
                                                                    loss=self.total_loss,
                                                                    optimizer="Adam",
                                                                    learning_rate=self.params['learning_rate'],
                                                                    #learning_rate_decay_fn=lambda lr, gs: tf.train.exponential_decay(params['learning_rate'], tf.train.get_global_step(), 100, 0.96, staircase=True),
                                                                    global_step=tf.train.get_global_step(),
                                                                    clip_gradients=5.0,
                                                                    summaries=["learning_rate", "global_gradient_norm"])

                        #with tf.variable_scope("stats"):
                            #nops = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
                            #tf.logging.info('Number of trainable parameters {}'.format(nops))


                    
                    self.eval_metrics = {}
                    mse, mse_update_op = tf.metrics.mean_squared_error(input_text_embeddings * mask, decoder_outputs_reconstructed * mask)

                    self.eval_metrics["mse"] = (mse, mse_update_op)