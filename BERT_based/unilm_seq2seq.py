import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import os

maxlen = 128 # most are under 100, we dont lose too much information
batch_size = 32
epochs = 50

config_path = './bert_config.json'
checkpoint_path = './bert_model.ckpt'
dict_path = './vocab.txt'

def load_data(filename):
  D = []
  with open(filename, 'r') as f:
    data = json.load(f)
    
  for item in data:
    D.append((item['rewrite_q'], item['new_q']))
  return D

total_data = load_data('data.json')


token_dict, keep_tokens = load_vocab(
	dict_path=dict_path, simplified=True,
	startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def filter_data(ori_data):
  new_data = []
  for target, bland in ori_data:
    tmp_ids, _ = tokenizer.encode(bland)
    if len(tmp_ids) * 2 < maxlen:
      new_data.append((target, bland))

  return new_data

total_data = filter_data(total_data)

train_data = total_data[:int(len(total_data) * 0.8)]
valid_data = total_data[int(len(total_data) * 0.8):]


class data_generator(DataGenerator):
	def __iter__(self, random=False):
		batch_token_ids, batch_segment_ids = [], []
		for is_end, (target, bland) in self.sample(random):
			token_ids, segment_ids = tokenizer.encode(bland, target, maxlen=maxlen)
			batch_token_ids.append(token_ids)
			batch_segment_ids.append(segment_ids)
			if len(batch_token_ids) == self.batch_size or is_end:
				batch_token_ids = sequence_padding(batch_token_ids)
				batch_segment_ids = sequence_padding(batch_segment_ids)
				yield [batch_token_ids, batch_segment_ids], None
				batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
	def compute_loss(self, inputs, mask=None):
		y_true, y_mask, y_pred = inputs
		y_true = y_true[:, 1:]
		y_mask = y_mask[:, 1:]
		y_pred = y_pred[:, :-1]
		loss = K.sparse_categorical_crossentropy(y_true, y_pred)
		loss = K.sum(loss * y_mask) / K.sum(y_mask)
		return loss


model = build_transformer_model(
	config_path, checkpoint_path,
	application='unilm', keep_tokens=keep_tokens)

output = CrossEntropy(2)(model.inputs + model.outputs)
model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
# model.summary()


class Seq2Seq(AutoRegressiveDecoder):
  @AutoRegressiveDecoder.wraps(default_rtype='probas')
  def predict(self, inputs, output_ids, states):
    token_ids, segment_ids = inputs
    token_ids = np.concatenate([token_ids, output_ids], 1)
    segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
    return model.predict([token_ids, segment_ids])[:, -1]
  
  def generate(self, text, topk=1):
    # print(maxlen, self.maxlen)
    max_c_len = maxlen - self.maxlen
    token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
    output_ids = self.beam_search([token_ids, segment_ids], topk)
    return tokenizer.decode(output_ids)


seq2seq_model = Seq2Seq(start_id=None, end_id=tokenizer._token_end_id, maxlen=64)


class Evaluator(keras.callbacks.Callback):
	def __init__(self):
		self.rouge = Rouge()
		self.smooth = SmoothingFunction().method1
		self.best_bleu = 0.0

	def on_epoch_end(self, epoch, logs=None):
		metrics = self.evaluate(valid_data)
		if metrics['bleu'] > self.best_bleu:
			self.best_bleu = metrics['bleu']
			model.save_weights('best_model_epoch_' + str(epoch) + '.weights')
		metrics['best_bleu'] = self.best_bleu
		print('valid_data:', metrics)

	def evaluate(self, data, topk=1):
		total = 1
		rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
		for target, bland in tqdm(data):
			# print(target, bland)
			total += 1
			target = ' '.join(target).lower()
			pred_target = ' '.join(seq2seq_model.generate(bland, topk)).lower()
			if pred_target.strip():
				scores = self.rouge.get_scores(hyps=pred_target, refs=target)
				rouge_1 += scores[0]['rouge-1']['f']
				rouge_2 += scores[0]['rouge-2']['f']
				rouge_l += scores[0]['rouge-l']['f']
				bleu += sentence_bleu(
					references=[target.split(' ')],
					hypothesis=pred_target.split(' '),
					smoothing_function=self.smooth)
		rouge_1 /= total
		rouge_2 /= total
		rouge_l /= total
		bleu /= total
		return {'rouge_1': rouge_1, 'rouge_2': rouge_2, 'rouge_l': rouge_l, 'bleu': bleu}


evaluator = Evaluator()
train_generator = data_generator(train_data, batch_size)


model.fit_generator(
	train_generator.forfit(),
	steps_per_epoch=len(train_generator),
	epochs=epochs,
	callbacks=[evaluator])

with open('unlim_bert_pretrain_no_visual.txt', 'w') as f:
  f.write('Train:\n')
  for target, bland in tqdm(train_data):
    f.write('input: ' + bland + '\n')
    f.write('target: ' + target + '\n')
    f.write('output: ' + seq2seq_model.generate(bland, 1) + '\n')
    f.write('\n')

  f.write('Test:\n')
  for target, bland in tqdm(valid_data):
    f.write('input: ' + bland + '\n')
    f.write('target: ' + target + '\n')
    f.write('output: ' + seq2seq_model.generate(bland, 1) + '\n')
    f.write('\n')


#  model.load_weights('./best_model.weights')