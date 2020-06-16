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


def general_preprocess(qs, gen_tokenizer=None):
  # transfer the q_data into tokens
  # the input questino should be the output question (longer)
  # without tokenizer, build it
  # return tokens and tokenizer

  qs = [preprocess_sentence(q) for q in qs]
  qs = ['[CLS] ' + q + ' [SEP]' for q in qs]

  print(qs[0])

  if not gen_tokenizer:
  	#build tokenizer
    gen_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',
                                                           lower=False)
    gen_tokenizer.fit_on_texts(qs)
    gen_tokenizer.word_index['<pad>'] = 0
    gen_tokenizer.index_word[0] = '<pad>'
  
  # max_len = max(all qs)
  tokens = gen_tokenizer.texts_to_sequences(qs)
  tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens,
                                                         padding='post')
  return tokens, gen_tokenizer