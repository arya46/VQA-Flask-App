""" Custom Layers:

*NB: The variable have been named as per the paper (wherever possible).*
*Code Ref: https://github.com/ritvikshrivastava/ADL_VQA_Tensorflow2*
*Weights Initialization Ref: https://stats.stackexchange.com/a/393012*
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D

class AttentionMaps(tf.keras.layers.Layer):
  """
  Given an image feature map V ∈ R(d×N), and the question representation Q ∈ R(d×T), 
  calculates the affinity matrix C ∈ R(T×N): C = tanh((QT)(Wb)V) ; 
  where Wb ∈ R(d×d) contains the weights. (Refer eqt (3) section 3.3).

  Given this affinity matrix C ∈ R(T×N), predicts image and question attention maps 
  (Refer eqt (4) section 3.3).

  Arguments:
    dim_k     : hidden attention dimention
    reg_value : Regularization value


  Inputs:
    image_feat,    V : shape (N,  d) or (49, dim_d)
    ques_feat,     Q : shape (T,  d) or (23, dim_d)

  Outputs:
    Image and Question attention maps viz:
    a) Hv = tanh(WvV + (WqQ)C) and
    b) Hq = tanh(WqQ + (WvV )CT)
  """
  def __init__(self, dim_k, reg_value, **kwargs):
    super(AttentionMaps, self).__init__(**kwargs)

    self.dim_k = dim_k
    self.reg_value = reg_value

    self.Wv = Dense(self.dim_k, activation=None,\
                        kernel_regularizer=tf.keras.regularizers.l2(self.reg_value),\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=2))
    self.Wq = Dense(self.dim_k, activation=None,\
                        kernel_regularizer=tf.keras.regularizers.l2(self.reg_value),\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=3))

  def call(self, image_feat, ques_feat):
    """
    The main logic of this layer.
    """  

    # Affinity Matrix C
    # (QT)(Wb)V 
    C = tf.matmul(ques_feat, tf.transpose(image_feat, perm=[0,2,1])) # [b, 23, 49]
    # tanh((QT)(Wb)V)
    C = tf.keras.activations.tanh(C) 

    # (Wv)V
    WvV = self.Wv(image_feat)                             # [b, 49, dim_k]
    # (Wq)Q
    WqQ = self.Wq(ques_feat)                              # [b, 23, dim_k]

    # ((Wq)Q)C
    WqQ_C = tf.matmul(tf.transpose(WqQ, perm=[0,2,1]), C) # [b, k, 49]
    WqQ_C = tf.transpose(WqQ_C, perm =[0,2,1])            # [b, 49, k]

    # ((Wv)V)CT                                           # [b, k, 23]
    WvV_C = tf.matmul(tf.transpose(WvV, perm=[0,2,1]), tf.transpose(C, perm=[0,2,1]))  
                        
    WvV_C = tf.transpose(WvV_C, perm =[0,2,1])            # [b, 23, k]

    #---------------image attention map------------------
    # We find "Hv = tanh((Wv)V + ((Wq)Q)C)" ; H_v shape [49, k]

    H_v = WvV + WqQ_C                                     # (Wv)V + ((Wq)Q)C
    H_v = tf.keras.activations.tanh(H_v)                  # tanh((Wv)V + ((Wq)Q)C) 

    #---------------question attention map---------------
    # We find "Hq = tanh((Wq)Q + ((Wv)V)CT)" ; H_q shape [23, k]

    H_q = WqQ + WvV_C                                     # (Wq)Q + ((Wv)V)CT
    H_q = tf.keras.activations.tanh(H_q)                  # tanh((Wq)Q + ((Wv)V)CT) 
        
    return [H_v, H_q]                                     # [b, 49, k], [b, 23, k]
  
  def get_config(self):
    """
    This method collects the input shape and other information about the layer.
    """
    config = {
        'dim_k': self.dim_k,
        'reg_value': self.reg_value
    }
    base_config = super(AttentionMaps, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class ContextVector(tf.keras.layers.Layer):
  """
  Method to find context vector of the image and text features
  (Refer eqt (4) and (5) section 3.3).
  
  Arguments:
    reg_value : Regularization value
    
  Inputs:
    image_feat V: image features, (49, d)
    ques_feat  Q: question features, (23, d)
    H_v: image attention map, (49, k)
    H_q: question attention map, (23, k)

  Outputs:
    Returns d-dimenstional context vector for image and question features
  """
  def __init__(self, reg_value, **kwargs):
    super(ContextVector, self).__init__(**kwargs)

    self.reg_value = reg_value

    self.w_hv = Dense(1, activation='softmax',\
                        kernel_regularizer=tf.keras.regularizers.l2(self.reg_value),\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=4))
    self.w_hq = Dense(1, activation='softmax',\
                        kernel_regularizer=tf.keras.regularizers.l2(self.reg_value),\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=5)) 
    

  def call(self, image_feat, ques_feat, H_v, H_q):
    """
    The main logic of this layer.
    """  
    # attention probabilities of each image region vn; a_v = softmax(wT_hv * H_v)
    a_v = self.w_hv(H_v)                               # [b, 49, 1]

    # attention probabilities of each word qt ;        a_q = softmax(wT_hq * H_q)
    a_q = self.w_hq(H_q)                               # [b, 23, 1]

    # context vector for image
    v = a_v * image_feat                               # [b, 49, dim_d]
    v = tf.reduce_sum(v, 1)                            # [b, dim_d]

    # context vector for question
    q = a_q * ques_feat                                # [b, 23, dim_d]
    q = tf.reduce_sum(q, 1)                            # [b, dim_d]


    return [v, q]

  def get_config(self):
    """
    This method collects the input shape and other information about the layer.
    """
    config = {
        'reg_value': self.reg_value
    }
    base_config = super(ContextVector, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class PhraseLevelFeatures(tf.keras.layers.Layer):
    """
    We compute the phrase features by applying 1-D convolution on the word embedding 
    vectors with filters of three window sizes: unigram, bigram and trigram.
    The word-level features Qw are appropriately 0-padded before feeding into bigram and 
    trigram convolutions to maintain the length of the sequence after convolution.
    Given the convolution result, we then apply max-pooling across different n-grams at each word
    location to obtain phrase-level features
    (Refer eqt (1) and (2) section 3.2).
      
    Arguments:
      dim_d: hidden dimension
      
    Inputs:
      word_feat Q : word level features of shape (23, dim_d)
      
    Outputs:
      Phrase level features of the question of shape (23, dim_d)
    """
    def __init__(self, dim_d, **kwargs):
      super().__init__(**kwargs)
      
      self.dim_d = dim_d
      
      self.conv_unigram = Conv1D(dim_d, kernel_size=1, strides=1,\
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=6)) 
      self.conv_bigram =  Conv1D(dim_d, kernel_size=2, strides=1, padding='same',\
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=7)) 
      self.conv_trigram = Conv1D(dim_d, kernel_size=3, strides=1, padding='same',\
                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=8)) 
    
    
    def call(self, word_feat):
      """
      The main logic of this layer.
    
      Compute the n-gram phrase embeddings (n=1,2,3)
      """
      # phrase level unigram features
      x_uni = self.conv_unigram(word_feat)                    # [b, 23, dim_d]
    
      # phrase level bigram features
      x_bi  = self.conv_bigram(word_feat)                     # [b, 23, dim_d]
    
      # phrase level trigram features
      x_tri = self.conv_trigram(word_feat)                    # [b, 23, dim_d]
    
      # Concat
      x = tf.concat([tf.expand_dims(x_uni, -1),\
                      tf.expand_dims(x_bi, -1),\
                      tf.expand_dims(x_tri, -1)], -1)         # [b, 23, dim_d, 3]
    
      # https://stackoverflow.com/a/36853403
      # Max-pool across n-gram features; over-all phrase level feature
      x = tf.reduce_max(x, -1)                                # [b, 23, dim_d]
      print(x)
      return x
    
    def get_config(self):
      """
      This method collects the input shape and other information about the layer.
      """
      config = {
          'dim_d': self.dim_d
      }
      base_config = super().get_config()
      return dict(list(base_config.items()) + list(config.items()))