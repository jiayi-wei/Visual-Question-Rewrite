import model
import tokenizer
import data_preprocess

import tensorflow as tf

import os
import time


###############
##   config  ##
root = "./"

exp_name = "res50_gen_token"

cate = "human_annot"
img_model_name = 'res50'
fc_top = False

#################
###  data read ##

print("data preprocessing")
# q_data longer, new_q_data shorter
q_data, new_q_data, img_data = data_preprocess.read_data(root, cate)

img_data = data_preprocess.extract_img_feat(root, 
                                            cate,
                                            img_data,
                                            name=img_model_name,
                                            fc=fc_top)

target_ids, gen_tokenizer = tokenizer.general_preprocess(q_data)
input_ids, gen_tokenizer = tokenizer.general_preprocess(new_q_data, gen_tokenizer)

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


#####################
#### model & optim ##

print("model build")
decoder = model.QRewriteModel(vocab_tar_size,
                              embedding_dim,
                              units,
                              batch_size)
# encoder = model.bert_encoder(input_max_length)
encoder = model.GenEncoder(vocab_tar_size,
                           embedding_dim,
                           units)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


def train_step(ids, targ, img):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(ids)
    
    dec_hidden = enc_hidden
    
    dec_input = tf.expand_dims([gen_tokenizer.word_index['[CLS]']] * batch_size, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output and image feature to the decoder
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


################
#### training ##

print("Begin training")
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  total_loss = 0

  for batch, (ids, targ, img) in enumerate(dataset):
    # enc_hidden, enc_output = encoder(ids, masks, segments)
    batch_loss = train_step(ids, targ, img)
    total_loss += batch_loss

    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                 batch,
                                                 batch_loss.numpy()))
    
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix=checkpoint_prefix)
  
  print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
