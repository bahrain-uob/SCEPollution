FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3
RUN pip3 install -U pip testresources setuptools
RUN apt update && apt install -y --fix-missing make g++
RUN apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran -y
RUN pip3 install pkgconfig
RUN apt update && apt install -y --fix-missing python3-pip libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev libjpeg8-dev liblapack-dev libblas-dev gfortran python3-h5py
RUN HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/ pip3 install pip testresources setuptools cython numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 pybind11
RUN pip3 install pandas
RUN pip3 install requests
RUN pip3 install tqdm 
RUN pip3 install PyYaml
RUN pip3 install matplotlib
RUN pip3 install opencv-python
RUN pip3 install scipy
RUN apt-get update
# RUN pip3 install numpy future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig 
# RUN env H5PY_SETUP_REQUIRES=0 pip3 install h5py
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
CMD ["python3", "/home/JetsonYolo/JetsonYolo.py"]