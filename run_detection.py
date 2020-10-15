import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

def get_keypoint_tuples(eval_config):
    """Return a tuple list of keypoint edges from the eval config.

    Args:
        eval_config: an eval config containing the keypoint edges

    Returns:
        a list of edge tuples, each in the format (start, end)
    """
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
        path: the file path to the image

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

model_name = 'centernet_hourglass104_1024x1024_coco17_tpu-32'

pipeline_config = os.path.join('object_detection/configs/tf2/', model_name + '.config')
model_dir = 'object_detection/test_data/checkpoint/'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
# print(configs)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn

detect_fn = get_model_detection_function(detection_model)

label_map_path = 'object_detection/data/mscoco_label_map.pbtxt'
# print(label_map_path)
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
# print(category_index)
# print(label_map_dict)

image_dir_list = ['auto_annot/image', 'human_annot/image']
visualization = False

for image_dir in image_dir_list:
    image_list = os.listdir(image_dir)
    det_res_dir = os.path.join(image_dir.split('/')[0], 'det_res')
    if not os.path.exists(det_res_dir):
        os.mkdir(det_res_dir)

    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name)
        image_np = load_image_into_numpy_array(image_path)

        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)
        
        f = open(os.path.join(det_res_dir, image_name.split('.')[0] + '.txt'), 'w')
        for i in range(detections['num_detections'][0].numpy()):
            if detections['detection_classes'][0][i].numpy() in category_index and detections['detection_scores'][0][i].numpy() >= 0.3:
                item = category_index[detections['detection_classes'][0][i].numpy()]['name']
                item += ' ' + str(detections['detection_scores'][0][i].numpy())
                for j in range(4):
                    item += ' ' + str(detections['detection_boxes'][0][i][j].numpy())
                item += '\n'
                f.write(item)

        f.close()
        # print("detections", detections)
        # print("predictions_dict", predictions_dict)
        # print("shapes", shapes)
        # print(input_tensor.shape)
        # quit()
        if visualization:
            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            keypoints, keypoint_scores = None, None

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False,
                keypoints=keypoints,
                keypoint_scores=keypoint_scores,
                keypoint_edges=get_keypoint_tuples(configs['eval_config']))

            plt.figure(figsize=(12,16))
            plt.imshow(image_np_with_detections)
            # plt.show()
            plt.savefig(os.path.join("object_detection/vis_det", image_to_det))
