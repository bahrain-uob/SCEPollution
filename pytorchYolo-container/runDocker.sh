#!/bin/bash
sudo docker build -t detector .
sudo xhost +
sudo docker run --device=/dev/ttyACM0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it --rm --runtime nvidia --network host -v /home/trafficpollution/repo/SCEPollution:/home -v /home/trafficpollution/repo/SCEPollution/cert:/home/cert detector  # <--- Here
