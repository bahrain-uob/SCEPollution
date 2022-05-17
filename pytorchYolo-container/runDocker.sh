#!/bin/bash
sudo docker build -t detector .
sudo xhost +
sudo docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it --rm --runtime nvidia --network host -v /home/trafficpollution/repo/SCEPollution:/homE detector  # <--- Here
