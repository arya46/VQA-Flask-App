from .layers import ContextVector, PhraseLevelFeatures, AttentionMaps
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Conv1D, Dropout, Input, LSTM, concatenate
from tensorflow.keras.models import Model

def build_model(max_answers, max_seq_len, vocab_size, dim_d, dim_k, l_rate, d_rate, reg_value):
    """
    Defines the Keras model.

    Arguments
    ----------
    max_answers : Number of output targets of the model.
    max_seq_len : Length of input sequences
    vocab_size  : Size of the vocabulary, i.e. maximum integer index + 1.
    dim_d       : Hidden dimension
    dim_k       : Hidden attention dimension
    l_rate      : Learning rate for the model
    d_rate      : Dropout rate
    reg_value   : Regularization value

    Returns
    ----------
    Returns the Keras model.
    """
    # inputs 
    image_input = Input(shape=(49, 512, ), name='Image_Input')
    ques_input = Input(shape=(22, ), name='Question_Input')

    # image feature; (Wb)V                                          # [b, 49, dim_d]
    image_feat = Dense(dim_d, activation=None, name='Image_Feat_Dense',\
                            kernel_regularizer=tf.keras.regularizers.l2(reg_value),\
                                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1))(image_input)
    image_feat = Dropout(d_rate, seed=1)(image_feat)

    # word level
    ques_feat_w = Embedding(input_dim=vocab_size, output_dim=dim_d, input_length=max_seq_len,\
                            mask_zero=True)(ques_input)
    
    Hv_w, Hq_w = AttentionMaps(dim_k, reg_value, name='AttentionMaps_Word')(image_feat, ques_feat_w)
    v_w, q_w = ContextVector(reg_value, name='ContextVector_Word')(image_feat, ques_feat_w, Hv_w, Hq_w)
    feat_w = tf.add(v_w,q_w)
    h_w = Dense(dim_d, activation='tanh', name='h_w_Dense',\
                    kernel_regularizer=tf.keras.regularizers.l2(reg_value),\
                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=13))(feat_w)

    # phrase level
    ques_feat_p = PhraseLevelFeatures(dim_d, name='PhraseLevelFeatures')(ques_feat_w)

    Hv_p, Hq_p = AttentionMaps(dim_k, reg_value, name='AttentionMaps_Phrase')(image_feat, ques_feat_p)
    v_p, q_p = ContextVector(reg_value, name='ContextVector_Phrase')(image_feat, ques_feat_p, Hv_p, Hq_p)
    feat_p = concatenate([tf.add(v_p,q_p), h_w], -1) 
    h_p = Dense(dim_d, activation='tanh', name='h_p_Dense',\
                    kernel_regularizer=tf.keras.regularizers.l2(reg_value),\
                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=14))(feat_p)

    # sentence level
    ques_feat_s = LSTM(dim_d, return_sequences=True, input_shape=(None, max_seq_len, dim_d),\
                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=16))(ques_feat_p)

    Hv_s, Hq_s = AttentionMaps(dim_k, reg_value, name='AttentionMaps_Sent')(image_feat, ques_feat_s)
    v_s, q_s = ContextVector(reg_value, name='ContextVector_Sent')(image_feat, ques_feat_p, Hv_s, Hq_s)
    feat_s = concatenate([tf.add(v_s,q_s), h_p], -1) 
    h_s = Dense(2*dim_d, activation='tanh', name='h_s_Dense',\
                    kernel_regularizer=tf.keras.regularizers.l2(reg_value),\
                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=15))(feat_s)

    z   = Dense(2*dim_d, activation='tanh', name='z_Dense',\
                    kernel_regularizer=tf.keras.regularizers.l2(reg_value),\
                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=16))(h_s)
    z   = Dropout(d_rate, seed=16)(z)

    # result
    result = Dense(max_answers, activation='softmax')(z)

    model = Model(inputs=[image_input, ques_input], outputs=result)

    return model