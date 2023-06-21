from detector import Detector

MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
CLASSES_FILE = "coco.names"
IMAGE_PATH = "tests/1.jpg"
VIDEO_PATH = "tests/1.mp4"
THRESHOLD = 0.5

detector = Detector()
detector.read_classes(CLASSES_FILE)
detector.download_model(MODEL_URL)
detector.load_model()
# detector.predict_image(IMAGE_PATH, THRESHOLD)
detector.predict_video(VIDEO_PATH, THRESHOLD)

