import os
import tensorflow as tf
import json
import numpy as np


def read_data(root, cate):
  # input cate = human_annot or auto_annot
  # root is the root path of data
  # read data into three lists
  # q: original q (long)
  # new_q: shorten q
  # img: img_path
  path = os.path.join(root, cate, 'data.json')
  q, new_q, img = [], [], []
  with open(path, 'r') as f:
    data = json.load(f)
  for item in data:
    q.append(item['rewrite_q'])
    new_q.append(item['new_q'])
    img.append(item['img_path'])
  return q, new_q, img


def get_model(name, fc):
  # name = vgg19 or res50
  # fc = True or False
  # return the pretrained model

  # choose the image model according to input config
  if name == 'vgg19' and not fc:
    model = tf.keras.applications.VGG19(include_top=False,
                                        weights='imagenet')
    output = model.layers[-1].output
  elif name == 'vgg19' and fc:
    model = tf.keras.applications.VGG19(include_top=True,
                                        weights='imagenet')
    output = model.layers[-3].output
  elif name == 'res50' and not fc:
    model = tf.keras.applications.ResNet50(include_top=False,
                                           weights='imagenet')
    output = model.layers[-1].output
  else:
    print("Illegal Config.")

  # build new image model
  input = model.input
  img_model = tf.keras.Model(input, output)
  return img_model


def feature_path(img_path, name, fc):
  # name is the name of image model, fc is true of false
  # "./drive/My Drive/questions_rewrite/auto_annot/image/000002.jpg"
  # if cate=res50 and fc=False
  # "./drive/My Drive/questions_rewrite/auto_annot/image/000002_res50_nonfc.npy""
  p, img_name = img_path.rsplit('/', 1)
  new_name = img_name.split('.')[0] + "_{}_{}.npy"
  fc_ = 'fc' if fc else 'nonfc'
  new_name = new_name.format(name, fc_)
  return os.path.join(p, new_name)


def extract_img_feat(root, cate, img_data, name='res50', fc=False, extract_feature=True):
  # input: cate = vgg19 or res50
  #		   img_data = list of img_data
  # set the image model accoding to the name and fc config
  if not extract_feature:
    for i in range(len(img_data)):
      img_data[i] = feature_path(os.path.join(root, cate, img_data[i]), name, fc)

    return img_data

  img_model = get_model(name, fc)

  # load img function
  def load_image(image_path):
  	# load image with TF
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    # modify image according to the image model
    if name == 'vgg19':
      img = tf.keras.applications.vgg19.preprocess_input(img)
    elif name == 'res50':
      img = tf.keras.applications.resnet.preprocess_input(img)
    else:
      print("Illegal Name.")
    return img, image_path

  # get unique images
  unique_img = list(set(img_data))
  unique_img = list(map(lambda x: os.path.join(root, cate, x), unique_img))

  # build dataset for unique images
  image_dataset = tf.data.Dataset.from_tensor_slices(unique_img)
  image_dataset = image_dataset.map(
      load_image,
      num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

  print("Model and Dataset Done.")

  # extract features and save
  for img, path in image_dataset:
      batch_features = img_model(img)
      # reshpae image features
      # vgg19 without top: [7*7, 512]
      # vgg19 with top: [4096]
      # res50 without top: [7*7, 2048]
      if not fc:
        batch_features = tf.reshape(batch_features, 
                                      (batch_features.shape[0],
                                      -1, batch_features.shape[-1]))
      for f, p in zip(batch_features, path):
        path_of_feature = feature_path(p.numpy().decode("utf-8"), name, fc)
        np.save(path_of_feature, f.numpy())

  print("Extraction Done.")

  # exchange img_data with feature path
  for i in range(len(img_data)):
    img_data[i] = feature_path(os.path.join(root, cate, img_data[i]), name, fc)
  
  return img_data


def train_val_split(data, rate=0.9):
  len_ = len(data)
  return data[:int(len_ * rate)], data[int(len_ * rate):]


def load_func_bert(ids, masks, segs, targ, img):
  img_tensor = np.load(img.decode('utf-8'))
  return ids, masks, segs, targ, img_tensor


def load_func_gen(ids, targ, img):
  img_tensor = np.load(img.decode('utf-8'))
  return ids, targ, img_tensor
