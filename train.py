import model
import tokenizer
import data_preprocess

import tensorflow as tf

import os
import time


###############
##   config  ##
root = "./"

exp_name = "res50_gen_token_2"

cate = "auto_annot"
# cate = "human_annot"
with_visual = False
# with_visual = True

# extract_feature = False
extract_feature = True

if with_visual:
    exp_name += "_visual_"
else:
    exp_name += "_no_visual_"

exp_name += cate


img_model_name = 'res50'
fc_top = False

########################
###  RNN part config ###
encoder_bidirection = True
if encoder_bidirection:
    exp_name += "_encoder_bidirect"
else:
    exp_name += "_no_encoder_bidirect"

# [1-2, 1-2]
GRU_layers = {"encoder": 2, "decoder": 2}
exp_name += "_GRU_layers_enc" + GRU_layers["encoder"] + "_dec" + GRU_layers["decoder"]

#################
###  data read ##

print("data preprocessing")
# q_data longer, new_q_data shorter

q_data, new_q_data, img_data = data_preprocess.read_data(root, cate)

target_ids, input_ids, img_data, gen_tokenizer = tokenizer.general_preprocess(q_data, new_q_data, img_data)
print(target_ids.shape)
print(input_ids.shape)
print(len(img_data))
# quit()

img_data = data_preprocess.extract_img_feat(root,
                                            cate,
                                            img_data,
                                            name=img_model_name,
                                            fc=fc_top,
                                            extract_feature=extract_feature)

# target_ids, gen_tokenizer = tokenizer.general_preprocess(q_data)
# input_ids, gen_tokenizer = tokenizer.general_preprocess(new_q_data, gen_tokenizer)

train_input_ids, val_input_ids = data_preprocess.train_val_split(input_ids)
# train_input_masks, val_input_masks = data_preprocess.train_val_split(input_masks)
# train_input_segments, val_input_segments = data_preprocess.train_val_split(input_segments)

train_target_ids, val_target_ids = data_preprocess.train_val_split(target_ids)

train_img, val_img = data_preprocess.train_val_split(img_data)

###################
## model config ###

print("dataset build")
buffer_size = len(train_input_ids)
batch_size = 64
steps_per_epoch = len(train_input_ids) // batch_size
vocab_tar_size = len(gen_tokenizer.word_index) + 1
embedding_dim = 256
units = 768
# with_visual = False
max_length_targ, max_length_inp = target_ids.shape[1], input_ids.shape[1]
####################
#### dataset #######
# bert input encoding
# dataset = tf.data.Dataset.from_tensor_slices((train_input_ids,
#                                              train_input_masks,
#                                              train_input_segments,
#                                              train_target_ids,
#                                              train_img))
# dataset = dataset.map(lambda ids, masks, segs, targ, img:
#                          tf.numpy_function(load_func,
#                                            [ids, masks, segs, targ, img],
#                                            [tf.int32, tf.int32, tf.int32, tf.int32, tf.float32]),
#                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

# general tokenize input and output
dataset = tf.data.Dataset.from_tensor_slices((train_input_ids,
                                              train_target_ids,
                                              train_img))
dataset = dataset.map(lambda ids, targ, img:
                          tf.numpy_function(data_preprocess.load_func_gen,
                                            [ids, targ, img],
                                            [tf.int32, tf.int32, tf.float32]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = dataset.shuffle(buffer_size).batch(batch_size)


# val dataset
val_dataset = tf.data.Dataset.from_tensor_slices((val_input_ids,
                                                  val_target_ids,
                                                  val_img))
val_dataset = val_dataset.map(lambda ids, targ, img:
                                  tf.numpy_function(data_preprocess.load_func_gen,
                                                    [ids, targ, img],
                                                    [tf.int32, tf.int32, tf.float32]),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(1)


#####################
#### model & optim ##

print("model build")
embedding_layer = tf.keras.layers.Embedding(vocab_tar_size, embedding_dim)
decoder = model.QRewriteModel(vocab_tar_size,
                              embedding_layer,
                              units,
                              batch_size,
                              with_visual=with_visual,
                              gru_layers=GRU_layers["decoder"])
# encoder = model.bert_encoder(input_max_length)
encoder = model.GenEncoder(embedding_layer,
                           units,
                           encoder_bidirection,
                           gru_layers=GRU_layers["encoder"])

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints/' + exp_name 
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


def train_step(ids, targ, img):
  loss = 0

  with tf.GradientTape() as tape:
    # ids    batch*len
    # enc_output   batch*len*units
    # enc_hidden   batch*units
    enc_output, enc_hidden = encoder(ids)

    dec_hidden = enc_hidden

    # dec_input   batch*1
    dec_input = tf.expand_dims([gen_tokenizer.word_index['[CLS]']] * ids.shape[0], 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output and image feature to the decoder
      # predictions, dec_hidden, _, _ = decoder(dec_input,
      #                                        dec_hidden,
      #                                        enc_output,
      #                                        img)

      # dec_input       batch*1
      # dec_hidden      batch*units
      # enc_output      batch*len*units
      predictions, dec_hidden, _, _ = decoder(dec_input,
                                           dec_hidden,
                                           enc_output,
                                           img)

      loss += loss_function(targ[:, t], predictions)
      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))
  return batch_loss


def eval(input_dataset, data_size, return_result=False):
  total_loss = 0

  if return_result == True:
    input_sentences = []
    target_sentences = []
    output_sentences = []
    all_text_weights = []
    all_img_weights = []

  for ids, targ, img in input_dataset:
    batch_loss = 0.0
    if return_result:
      # inp_sen = ""
      inp_sen = gen_tokenizer.sequences_to_texts(ids.numpy())[0].replace(' <pad>', '')
      tar_sen = gen_tokenizer.sequences_to_texts(targ.numpy())[0].replace(' <pad>', '')
      out_sen = ""
      text_weights = []
      img_weights = []
    
    enc_output, enc_hidden = encoder(ids)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([gen_tokenizer.word_index['[CLS]']] * ids.shape[0], 1)
    
    for t in range(1, max_length_targ):
      predictions, dec_hidden, text_weight, img_weight = decoder(dec_input, dec_hidden, enc_output, img)
      
      batch_loss += loss_function(targ[:, t], predictions)
      
      pre_ = tf.argmax(predictions[0]).numpy()
      pre_ids = gen_tokenizer.index_word[pre_]
      
      if return_result == True:
        out_sen += pre_ids + ' '
        text_weights.append(text_weight.numpy())
        if img_weights:
            img_weights.append(img_weight.numpy())

      if pre_ids == "[SEP]":
        break

      dec_input = tf.expand_dims([pre_], 1)

    # total_loss = total_loss / data_size
    
    batch_loss = batch_loss / int(targ.shape[1])
    total_loss += batch_loss

    if return_result == True:
      input_sentences.append(inp_sen)
      target_sentences.append(tar_sen)
      output_sentences.append(out_sen)
      all_text_weights.append(text_weights)
      all_img_weights.append(img_weights)
  
  total_loss = total_loss / data_size

  if return_result:
    return total_loss, input_sentences, target_sentences, output_sentences, all_text_weights, all_img_weights

  return total_loss

################
#### training ##

print("Begin training")
EPOCHS = 50
eval_loss = 100.0

for epoch in range(EPOCHS):
  start = time.time()

  # enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for batch, (ids, targ, img) in enumerate(dataset):
    # enc_hidden, enc_output = encoder(ids, masks, segments)
    batch_loss = train_step(ids, targ, img)
    total_loss += batch_loss

    if batch % 10 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  eval_loss_after_epoch = eval(val_dataset, len(val_target_ids), return_result=False)

  if eval_loss > eval_loss_after_epoch:
    eval_loss = eval_loss_after_epoch
    print("Model Saved with eval loss {}".format(eval_loss))
    checkpoint.save(file_prefix=checkpoint_prefix+"_eval")
  elif (epoch + 1) % 10 == 0:
    print("Model Saved")
    checkpoint.save(file_prefix=checkpoint_prefix+"_epoch")

  print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# evaluate after training
print("begin eval")
train_dataset = tf.data.Dataset.from_tensor_slices((train_input_ids,
                                                    train_target_ids,
                                                    train_img))
train_dataset = train_dataset.map(lambda ids, targ, img:
                          tf.numpy_function(data_preprocess.load_func_gen,
                                            [ids, targ, img],
                                            [tf.int32, tf.int32, tf.float32]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_dataset = train_dataset.batch(1)

train_loss, train_input, train_target, train_output, train_text_w, train_img_w = eval(train_dataset, len(train_target_ids), return_result=True)

val_loss, val_input, val_target, val_output, val_text_w, val_img_w = eval(val_dataset, len(val_target_ids), return_result=True)

f = open(os.path.join(checkpoint_dir, "results_best_on_train.txt"), 'w')
f.write("Train\n\n")
for i in range(len(train_input)):
  f.write(train_input[i] + '\n')
  f.write(train_target[i] + '\n')
  f.write(train_output[i] + '\n\n')
f.write("Eval\n\n")
for i in range(len(val_input)):
  f.write(val_input[i] + '\n')
  f.write(val_target[i] + '\n')
  f.write(val_output[i] + '\n\n')
f.close()

'''
# restore from best model on val
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

train_loss, train_input, train_target, train_output, train_text_w, train_img_w = eval(train_dataset, len(train_target_ids), return_result=True)

val_loss, val_input, val_target, val_output, val_text_w, val_img_w = eval(val_dataset, len(val_target_ids), return_result=True)

f = open(os.path.join(checkpoint_dir, "results_best_on_eval.txt"), 'w')
f.write("Train\n\n")
for i in range(len(train_input)):
  f.write(train_input[i] + '\n')
  f.write(train_target[i] + '\n')
  f.write(train_output[i] + '\n\n')
f.write("Eval\n\n")
for i in range(len(val_input)):
  f.write(val_input[i] + '\n')
  f.write(val_target[i] + '\n')
  f.write(val_output[i] + '\n\n')
f.close()
'''
