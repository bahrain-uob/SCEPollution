import time
import cv2
import numpy as np
from elements.yolo import OBJ_DETECTION
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from application_util import preprocessing
from application_util import visualization

Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush' ]

Object_colors = list(np.random.rand(80,3)*255)
Object_detector = OBJ_DETECTION('weights/yolov5s.pt', Object_classes)

# Definition of the parameters
max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0
# initialize deep sort
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)

# begin video capture
video_path="./cars.mp4"
try:
    vid = cv2.VideoCapture(int(video_path))
except:
    vid = cv2.VideoCapture(video_path)
out = None

print('start object detection')
# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
while vid.isOpened():
    return_value, frame = vid.read()
    if return_value:
        start_time = time.time()
        detections = Object_detector.detect(frame)
        scores = np.array([d.score for d in detections])
        # boxes = 
        for obj in detections:
            # print(obj)
            label = obj['label']
            score = obj['score']
            # [(xmin,ymin),(xmax,ymax)] = obj['bbox']
            print(obj['bbox'])
            color = Object_colors[Object_classes.index(label)]
            # frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
            # frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA)
            # print(label)
        # end_time=time.time()-start_time
        # print("Inference time = " + str(end_time))
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
    else:
        print('Video has ended or failed, try a different video format!')
        break
    # cv2.imshow("Video", frame)
    keyCode = cv2.waitKey(30)
    if keyCode == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
