
# Run YoloV4 tiny with CLI only

python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/cars.mp4 --output ./outputs/tiny.avi --tiny --count --info --dont_show

# Convert TRT

python3.6 convert_trt.py --weights ./checkpoints/yolov4-tiny-416 

# Run the optimized tensorRT version 
python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/cars.mp4 --output ./outputs/tiny.avi --tiny --count --info --dont_show


 # nvidia container
 docker pull nvcr.io/nvidia/tensorflow:22.03-tf2-py3