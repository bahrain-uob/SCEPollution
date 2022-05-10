#!/bin/bash
sudo docker build -t detector .
sudo docker run -it --rm --runtime nvidia --network host -v /home/trafficpollution/repo/SCEPollution:/home detector -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  # <--- Here
