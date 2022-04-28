FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3
RUN apt-get update -y
RUN pip3 install pip -U testresources setuptools
RUN pip3 install requests
RUN pip3 install tqdm 
RUN pip3 install PyYaml
RUN pip3 install matplotlib
RUN pip3 install opencv-python
RUN pip3 install scipy
RUN pip3 install pandas

# attempts to build an image from two base images
FROM nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3

# install scipy and associated dependencies
# RUN apt-get install -y libblas3 liblapack3 liblapack-dev libblas-dev
# RUN apt-get install -y gfortran
# RUN apt-get install -y libatlas-base-dev
# RUN apt-get install -y python3-scipy
# RUN apt-get install -y wget
# RUN apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
# RUN pip3 install keras_applications==1.0.8 --no-deps
# RUN pip3 install keras_preprocessing==1.1.0 --no-deps
# RUN apt-get install -y python3-h5py
# RUN apt-get install -y openmpi-bin 
# RUN apt-get install -y libopenmpi-dev
# RUN apt-get install -y libatlas-base-dev
# RUN pip3 install six==1.15.0
# RUN pip3 install wheel==0.36.2
# RUN pip3 install mock==4.0.3
# RUN wget -q https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl
# RUN pip3 install tensorflow-2.4.0-cp37-none-linux_armv7l.whl --no-deps

# RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
CMD ["python3", "/home/JetsonYolo/JetsonYolo.py"]