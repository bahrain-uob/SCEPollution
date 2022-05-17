import time
import cv2
import numpy as np
# import serial
from elements.yolo import OBJ_DETECTION
# deep sort imports
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from application_util import preprocessing
from application_util import visualization
from tools import generate_detections as gdet
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from scipy.optimize import linear_sum_assignment as linear_assignment

from IoT_helper import helper
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import logging
import argparse
import json

AllowedActions = ['both', 'publish', 'subscribe']

# MQTT parsing arguments
# Read in command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--endpoint", action="store", required=True, dest="host", help="Your AWS IoT custom endpoint")
parser.add_argument("-r", "--rootCA", action="store", required=True, dest="rootCAPath", help="Root CA file path")
parser.add_argument("-c", "--cert", action="store", dest="certificatePath", help="Certificate file path")
parser.add_argument("-k", "--key", action="store", dest="privateKeyPath", help="Private key file path")
parser.add_argument("-p", "--port", action="store", dest="port", type=int, help="Port number override")
parser.add_argument("-w", "--websocket", action="store_true", dest="useWebsocket", default=False,
                    help="Use MQTT over WebSocket")
parser.add_argument("-id", "--clientId", action="store", dest="clientId", default="basicPubSub",
                    help="Targeted client id")
parser.add_argument("-t", "--topic", action="store", dest="topic", default="sdk/test/Python", help="Targeted topic")
parser.add_argument("-m", "--mode", action="store", dest="mode", default="both",
                    help="Operation modes: %s"%str(AllowedActions))
parser.add_argument("-M", "--message", action="store", dest="message", default="Hello World!",
                    help="Message to publish")


args = parser.parse_args()
host = args.host
rootCAPath = args.rootCAPath
certificatePath = args.certificatePath
privateKeyPath = args.privateKeyPath
port = args.port
useWebsocket = args.useWebsocket
clientId = args.clientId
topic = args.topic

if args.mode not in AllowedActions:
    parser.error("Unknown --mode option %s. Must be one of %s" % (args.mode, str(AllowedActions)))
    exit(2)

if args.useWebsocket and args.certificatePath and args.privateKeyPath:
    parser.error("X.509 cert authentication and WebSocket are mutual exclusive. Please pick one.")
    exit(2)

if not args.useWebsocket and (not args.certificatePath or not args.privateKeyPath):
    parser.error("Missing credentials for authentication.")
    exit(2)
# Port defaults
if args.useWebsocket and not args.port:  # When no port override for WebSocket, default to 443
    port = 443
if not args.useWebsocket and not args.port:  # When no port override for non-WebSocket, default to 8883
    port = 8883

# Configure logging
logger = logging.getLogger("AWSIoTPythonSDK.core")
logger.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)

# Init AWSIoTMQTTClient
myAWSIoTMQTTClient = None
if useWebsocket:
    myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId, useWebsocket=True)
    myAWSIoTMQTTClient.configureEndpoint(host, port)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath)
else:
    myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId)
    myAWSIoTMQTTClient.configureEndpoint(host, port)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath, privateKeyPath, certificatePath)

helper.setDefaults(myAWSIoTMQTTClient)
print("Connecting to MQTT..")
myAWSIoTMQTTClient.connect()




#OBJECT TRACKING AND DETECTION

Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush']

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
allowed_classes = ['car', 'motorbike', 'bus', 'truck']

fpsCounter = 0
fpsSum = 0
total_start_time = 0
start = True
wait_frame_count = {}
WarmUpCount = 0 
fpscv2 = vid.get(cv2.CAP_PROP_FPS)

# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
while vid.isOpened():
    return_value, frame = vid.read()
    if return_value:
        start_time = time.time()
        detections = Object_detector.detect(frame)
        if WarmUpCount < 6:
            WarmUpCount += 1 
            continue
        if start:
            total_start_time = time.time() 
            start = False
        boxes = [] 
        labels = []
        scores = []

        for obj in detections:
            if obj['label'] in allowed_classes and obj['score']>0.5:
                [(xmin,ymin),(xmax,ymax)] = obj['bbox']
                h = ymax - ymin
                w = xmax - xmin
                boxes.append((xmin,ymin, w, h))
                labels.append(obj['label'])
                scores.append(obj['score'])

        boxes = np.array(boxes)
        labels = np.array(labels)
        scores = np.array(scores)

        # Start non max supression 
        features = encoder(frame, boxes)
        detections = [Detection(box, score, label, feature) for box, score, label, feature in zip(boxes, scores, labels, features)]
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
       # run non-maxima supression
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = np.array([detections[i] for i in indices])       
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        cars = 0
        trucks = 0
        busses = 0
        print('Objects being tracked: {}'.format(str(len(tracker.tracks))))        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            if class_name == 'car':
                cars += 1
            elif class_name == 'truck':
                trucks += 1
            elif class_name == 'bus':
                busses += 1

            #counting frames for each track
            trackID=str(track.track_id)
            if trackID in wait_frame_count.keys():
                wait_frame_count[trackID] += 1   
            else:
                wait_frame_count[trackID] = 1
            

            # output tracker information
            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            # visualize 
            color = Object_colors[int(track.track_id) % len(Object_colors)]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        
        print('At the current frame: {} cars, {} busses, {} trucks'.format(cars, busses, trucks))
        cv2.imshow("output", frame)
            # frame = cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2) 
            # frame = cv2.putText(frame, f'{label} ({str(score)})', (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX , 0.75, color, 1, cv2.LINE_AA)
        fps = 1.0 / (time.time() - start_time)
        fpsSum += fps
        fpsCounter += 1
        print("FPS: %.2f" % fps)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    # timing in seconds 
    elif time.time() - total_start_time  > 60: 
        current = time.time()
        # Duration in minutes 
        diff = (current - total_start_time) / 60
        fps_average = fpsSum / fpsCounter
        print('FPS average for {} min = {} fps'.format(str(diff), str(fps_average)))
        
        print(wait_frame_count)
        wait_time_count = {}
        for k in wait_frame_count:                
            waittime = wait_frame_count[k] / 29.73
            if waittime > 0:
                wait_time_count[k]  = waittime
        print(wait_time_count)
        
        # average wait time
        average_wait_time = sum(wait_time_count.values()) / len(wait_time_count)
        print('The average wait time is: {}'.format(str(average_wait_time)))
        print(f"fps: {fpscv2}")
        # PUBLISH TO MQTTT
        myAWSIoTMQTTClient.publish(topic, 'TEST', 1)

        break
    else:
        print('Restarting the video')
        # break
        vid = cv2.VideoCapture(video_path)

vid.release()
cv2.destroyAllWindows()

