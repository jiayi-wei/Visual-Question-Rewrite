protoc object_detection/protos/*.proto --python_out=.
python -m pip install .

wget http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_1024x1024_coco17_tpu-8.tar.gz
tar -xf centernet_hg104_1024x1024_coco17_tpu-8.tar.gz
mv centernet_hg104_1024x1024_coco17_tpu-8/checkpoint object_detection/test_data/
rm centernet_hg104_1024x1024_coco17_tpu-8.tar.gz
rm -r centernet_hg104_1024x1024_coco17_tpu-8/