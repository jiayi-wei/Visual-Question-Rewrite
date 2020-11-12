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
import cv2


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
    D.append((item['rewrite_q'], item['new_q'], item['img_path']))
  return D


token_dict, keep_tokens = load_vocab(
	dict_path=dict_path, simplified=True,
	startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


total_data = load_data('data.json')

def filter_data(ori_data):
  new_data = []
  for target, bland, image_path in ori_data:
    tmp_ids, _ = tokenizer.encode(bland)
    if len(tmp_ids) * 2 < maxlen:
      new_data.append((target, bland, image_path))

  return new_data

total_data = filter_data(total_data)

train_data = total_data[:int(len(total_data) * 0.8)]
valid_data = total_data[int(len(total_data) * 0.8):]


MobileNetV2 = keras.applications.mobilenet_v2.MobileNetV2
preprocess_input = keras.applications.mobilenet_v2.preprocess_input
image_model = MobileNetV2(include_top=False, pooling='avg')
img_size = 299


def read_image(f):
    """单图读取函数（对非方形的图片进行白色填充，使其变为方形）
    """
    img = cv2.imread(f)
    height, width = img.shape[:2]
    if height > width:
        height, width = img_size, width * img_size // height
        img = cv2.resize(img, (width, height))
        delta = (height - width) // 2
        img = cv2.copyMakeBorder(
            img,
            top=0,
            bottom=0,
            left=delta,
            right=height - width - delta,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    else:
        height, width = height * img_size // width, img_size
        img = cv2.resize(img, (width, height))
        delta = (width - height) // 2
        img = cv2.copyMakeBorder(
            img,
            top=delta,
            bottom=width - height - delta,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    img = img.astype('float32')
    return img[..., ::-1]  # cv2的读取模式为BGR，但keras的模型要求为RGB


class data_generator(DataGenerator):
  def __iter__ (self, random=False):
    batch_images, batch_token_ids, batch_segment_ids = [], [], []
    for is_end, (target, bland, image_path) in self.sample(random):
      batch_images.append(read_image(os.path.jion('../auto_annot/', image_path)))
      token_ids, segment_ids = tokenizer.encode(bland, target, maxlen=maxlen)
      batch_token_ids.append(token_ids)
      batch_segment_ids.append(segment_ids)
      if len(batch_token_ids) == self.batch_size or is_end:
        batch_token_ids = sequence_padding(batch_token_ids)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        yield [batch_token_ids, batch_segment_ids, batch_images], None
        batch_images, batch_token_ids, batch_segment_ids = [], [], []



class CrossEntropy(Loss):
  def compute_loss(self, inputs, mask=None):
    print(inputs)
    y_true, y_mask, y_pred = inputs
    y_true = y_true[:, 1:]
    y_mask = y_mask[:, 1:]
    y_pred = y_pred[:, :-1]
    loss = K.sparse_categorical_crossentropy(y_true, y_pred)
    loss = K.sum(loss * y_mask) / K.sum(y_mask)
    return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,
    layer_norm_cond=image_model.output,
    layer_norm_cond_hidden_size=128,
    layer_norm_cond_hidden_act='swish',
    additional_input_layers=image_model.input,
)

output = CrossEntropy(2)([model.inputs[0], model.inputs[1], model.outputs[0]])
model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))


class Seq2SeqWImg(AutoRegressiveDecoder):
  @AutoRegressiveDecoder.wraps(default_rtype='probas')
  def predict(self, inputs, output_ids, states):
    token_ids, segment_ids, image = inputs
    token_ids = np.concatenate([token_ids, output_ids], 1)
    segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
    return mode.predict([token_ids, segment_ids, image])[:, -1]

  def generate(self, text, image, topk=1):
    if is_string(image):
      image = read_image(image)
    max_c_len = maxlen - self.maxlen
    token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
    output_ids = self.beam_search([token_ids, segment_ids, image], topk)
    return tokenizer.decode(output_ids)


seq2seq_model = Seq2SeqWImg(start_id=None, end_id=tokenizer._token_end_id, maxlen=80)


class Evaluator(keras.callbacks.Callback):
  def __init__(self):
    self.rouge = Rouge()
    self.smooth = SmoothingFunction().method1
    self.best_bleu =  0.0

  def on_epoch_end(self, epoch, logs=None):
    metrics = self.evaluate(valid_data)
    if metrics['bleu'] > self.best_bleu:
      self.best_bleu = metrics['bleu']
      mode.save_weights('best_model_epoch_' + str(epoch) + '.weights')
    metrics['best_bleu'] = self.best_bleu
    print('valid_data: ', metrics)

  def evaluate(self, data, topk=1):
    total = 1
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    for target, bland, image_path in tqdm(data):
      total += 1
      target = ' '.join(target).lower()
      pred_target = ' '.join(seq2seq_model.generate(bland, image_path, topk)).lower()
      if pred_target.strip():
        scores = self.rouge.get_scores(hyps=pred_target, refs=target)
        rouge_1 += scores[0]['rouge-1']['f']
        rouge_2 += scores[0]['rouge-2']['f']
        rouge_l += scores[0]['rouge-l']['f']
        bleu += sentence_bleu(
            reference=[target.split(' ')],
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
    callbacks=[evaluator]
)