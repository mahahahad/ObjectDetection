from detector import Detector

# Faster R-CNN Model from TensorFlow's Object Detection Model Zoo
MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8.tar.gz"
CLASSES_FILE = "coco.names"
VIDEO_PATH = "tests/1.mp4"
THRESHOLD = 0.5

detector = Detector()
detector.read_classes(CLASSES_FILE)
detector.download_model(MODEL_URL)
detector.load_model()
detector.predict_video(VIDEO_PATH, THRESHOLD)

