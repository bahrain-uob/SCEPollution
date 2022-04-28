from cProfile import label
import time
import cv2
import numpy as np
from elements.yolo import OBJ_DETECTION
# deep sort imports
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from application_util import preprocessing
from application_util import visualization
from tools import generate_detections as gdet
from scipy.optimize import linear_sum_assignment as linear_assignment

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
Object_detector = OBJ_DETECTION('/home/JetsonYolo/weights/yolov5s.pt', Object_classes)

# Definition of the parameters
max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0
# initialize deep sort
# calculate cosine distance metric
model_filename = '/home/JetsonYolo/model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)

# begin video capture
video_path="/home/JetsonYolo/cars.mp4"
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
        print(detections)
        scores = np.array([d['score'] for d in detections])
        # turn (xmin,ymin), (xmax, ymax) to (x,y,width, height)
        boxes = [] 
        labels = []
        for obj in detections:
            [(xmin,ymin),(xmax,ymax)] = obj['bbox']
            h = ymax - ymin
            w = xmax - xmin
            boxes.append((xmin,ymin, w, h))
        boxes = np.array(boxes)
        # Start non max supression 
        features = encoder(frame, boxes)
        detections = [Detection(box, score, feature) for box, score,  feature in zip(boxes, scores, features)]
       # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
 
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        
        detections = np.array([detections[i] for i in indices])       
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            # output tracker information
            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

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
