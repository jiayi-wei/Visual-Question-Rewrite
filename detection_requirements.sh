protoc object_detection/protos/*.proto --python_out=.
python -m pip install .

wget http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_coco17_tpu-8.tar.gz
tar -xf centernet_hg104_512x512_coco17_tpu-8.tar.gz
mv centernet_hg104_512x512_coco17_tpu-8/checkpoint object_detection/test_data/
