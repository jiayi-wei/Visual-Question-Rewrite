import tensorflow as tf
import tensorflow_hub as hub

import bert

import re


######################
##  general functs  ##
######################
def preprocess_sentence(w):
  # transfer sentens with better format
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
  w = w.strip()
  # w = "[CLS] " + w + " [SEP]"
  return w



######################
##  bert tokenizer  ##
######################

FullTokenizer = bert.bert_tokenization.FullTokenizer


bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
bert_tokenizer = FullTokenizer(vocab_file, do_lower_case)


def get_masks(tokens, max_seq_length):
  if len(tokens) > max_seq_length:
    raise IndexError("Token longer than max_seq_length.")
  return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
  if len(tokens) > max_seq_length:
    raise IndexError("Token longer than max_seq_length.")
  segments = []
  current_segment_id = 0
  for token in tokens:
    segments.append(current_segment_id)
    if token == "[SEP]":
      current_segment_id = 1
  return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
  token_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
  return input_ids


def bert_preprocess(qs, tokenizer):
   max_length = 0
   qs_tokens = []
   for q in qs:
     q_data = preprocess_sentence(q)
     tokens = tokenizer.tokenize(q)
     tokens = ["[CLS]"] + tokens + ["[SEP]"]
     max_length = max(max_length, len(tokens))
     qs_tokens.append(tokens)

   ids = []
   masks = []
   segments = []
   for q_tokens in qs_tokens:
     ids.append(get_ids(q_tokens, tokenizer, max_length))
     masks.append(get_masks(q_tokens, max_length))
     segments.append(get_segments(q_tokens, max_length))
   return ids, masks, segments, max_length


# input_ids, input_masks, input_segments, input_max_length = bert_preprocess(new_q_data, bert_tokenizer)
# print(len(input_ids), len(input_ids[0]))
# 241 31

######################
# general tokenizer  #
######################


def general_preprocess(q_data, input_q_data, img_data):
  # transfer the q_data into tokens
  # the input questino should be the output question (longer)
  # without tokenizer, build it
  # return tokens and tokenizer

  q_data = [preprocess_sentence(q) for q in q_data]
  q_data = ['[CLS] ' + q + ' [SEP]' for q in q_data]

  input_q_data = [preprocess_sentence(q) for q in input_q_data]
  input_q_data = ['[CLS] ' + q + ' [SEP]' for q in input_q_data]

  print(q_data[0])
  print(input_q_data[0])

  #build tokenizer
  gen_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',
                                                           lower=False)
  gen_tokenizer.fit_on_texts(q_data)
  gen_tokenizer.word_index['<pad>'] = 0
  gen_tokenizer.index_word[0] = '<pad>'
  
  # max_len = max(all qs)
  tokens_q_data = gen_tokenizer.texts_to_sequences(q_data)

  q_data_ = []
  input_q_data_ = []
  img_data_ = []
  for i in range(len(tokens_q_data)):
    k = len(tokens_q_data[i]) // 10
    if k <= 4:
      q_data_.append(q_data[i])
      input_q_data.append(input_q_data[i])
      img_data_.append(img_data[i])

  tokens_q_data = gen_tokenizer.texts_to_sequences(q_data_)
  tokens_q_data = tf.keras.preprocessing.sequence.pad_sequences(tokens_q_data,
                                                         padding='post')
  tokens_input_q_data = gen_tokenizer.texts_to_sequences(input_q_data_)
  tokens_input_q_data = tf.keras.preprocessing.sequence.pad_sequences(tokens_input_q_data,
                                                              padding='post')
  return tokens_q_data, tokens_input_q_data, img_data_, gen_tokenizer