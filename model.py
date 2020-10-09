import tensorflow as tf
import tensorflow_hub as hub

import bert


class TextAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(TextAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query         batch*units
    # values        batch*len*units
    query_with_time_axis = tf.expand_dims(query, 1)
    # query_with_time_axis      batch*1*units
    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
    # score         batch*len*1
    attention_weights = tf.nn.softmax(score, axis=1)
    # attention_weights     batch*len  (probability)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    # context_vector        batch*units

    return context_vector, attention_weights


class ImageAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(ImageAttention, self).__init__()
    self.W_I = tf.keras.layers.Dense(units)
    self.W_Q = tf.keras.layers.Dense(units)
    self.W_P = tf.keras.layers.Dense(1)

  def call(self, v_I, v_Q):
    v_I_att = self.W_I(v_I)
    v_Q_att = self.W_Q(v_Q)
    v_Q_att = tf.expand_dims(v_Q_att, axis=1)
    p_I = self.W_P(tf.tanh(v_I_att + v_Q_att))
    attention_weights = tf.nn.softmax(p_I, axis=1)

    v_att = p_I * v_I
    v_att = tf.reduce_sum(v_att, axis=1)

    return v_att, attention_weights


class QRewriteModel(tf.keras.Model):
  def __init__(self,
               vocab_tar_size,
               # embedding_dim,
               embedding,
               dec_units,
               batch_size,
               with_visual=False):
    super(QRewriteModel, self).__init__()
    # self.batch_size = batch_size
    self.dec_units = dec_units
    self.with_visual = with_visual
    # self.dec_embedding = tf.keras.layers.Embedding(vocab_tar_size, embedding_dim)
    self.dec_embedding = embedding
    # return both sequence and state in GRU unit
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    # fc for classification
    self.fc = tf.keras.layers.Dense(vocab_tar_size)

    # self.encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
    #                              trainable=False)
    # two attention layers for both Image and Text
    self.text_attention = TextAttention(self.dec_units)
    self.img_attention = ImageAttention(self.dec_units)
    self.img_fc = tf.keras.layers.Dense(self.dec_units)
    if self.with_visual:
        self.img_attention = ImageAttention(self.dec_units)
        self.img_fc = tf.keras.layers.Dense(self.dec_units)

  def call(self, x, hidden, enc_output, img_feature=None):
    # get text feature based on encode output and last hidden
    # print(x.shape, hidden.shape, enc_output.shape)
    # x             batch*1
    # hidden        batch*units
    # enc_output    batch*len*units
    text_vector, text_weights = self.text_attention(hidden, enc_output)
    # text_vector   batch*units

    # get image feature based on encode output and last hidden
    if self.with_visual:
        img_feature = self.img_fc(img_feature)
        image_vector, image_weights = self.img_attention(img_feature, hidden)

    # x is the last predicted token.
    x = self.dec_embedding(x)
    # x             batch*embed

    # get the next state and seq
    # print(text_vector.shape)
    # print(x.shape)
    if self.with_visual:
        x = tf.concat([tf.expand_dims(text_vector, 1),
                       tf.expand_dims(image_vector, 1),
                       x], axis=-1)
    else:
        x = tf.concat([tf.expand_dims(text_vector, 1),
                       x], axis=-1)

    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))

    x = self.fc(output)

    if self.with_visual:
        return x, state, text_weights, image_weights
    return x, state, text_weights, None


class GenEncoder(tf.keras.Model):
  def __init__(self,
  	           # vocab_inp_size,
  	           # embedding_dim,
               embedding,
  	           enc_units,
               bidirection=False):
    super(GenEncoder, self).__init__()
    #self.enc_embedding = tf.keras.layers.Embedding(vocab_inp_size, embedding_dim)
    self.enc_units = enc_units

    self.enc_embedding = embedding
    if bidirection:
        self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            enc_units, return_state=True, return_sequences=True, recurrent_initializer='glorot_uniform'))
    else:
        self.gru = tf.keras.layers.GRU(
          enc_units, return_state=True, return_sequences=True, recurrent_initializer='glorot_uniform')

  def call(self, x):
      # x   batch*len
      x = self.enc_embedding(x)
      # x   batch*len*256
      # print(x.shape)
      # print(self.initialize_hidden_state(x.shape[0]).shape)
      output, state = self.gru(x, self.initialize_hidden_state(x.shape[0]))
      # output   batch*len*unit
      # state    batch*unit
      return output, state

  def initialize_hidden_state(self, batch_sz):
    return tf.zeros((batch_sz, self.enc_units))



def bert_encoder(input_max_length):
  # build the pre-trained bert encoder
  bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=True)

  input_word_ids = tf.keras.layers.Input(shape=(input_max_length,),
                                       dtype=tf.int32,
                                       name="input_word_ids")
  input_mask = tf.keras.layers.Input(shape=(input_max_length,),
                                   dtype=tf.int32,
                                   name="input_mask")
  segment_ids = tf.keras.layers.Input(shape=(input_max_length,),
                                    dtype=tf.int32,
                                    name="segment_ids")
  pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
  encoder = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
  return encoder
