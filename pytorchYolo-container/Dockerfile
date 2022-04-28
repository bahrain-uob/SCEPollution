FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3
RUN pip3 install -U pip testresources setuptools==49.6.0
RUN pip3 install pandas
RUN pip3 install requests
RUN pip3 install tqdm 
RUN pip3 install PyYaml
RUN pip3 install matplotlib
RUN pip3 install opencv-python
RUN pip3 install scipy
RUN apt-get update
RUN apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran -y
# RUN pip3 install -U pip testresources setuptools==49.6.0
RUN pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig 
RUN pip3 install pkgconfig
RUN env H5PY_SETUP_REQUIRES=0 pip3 install h5py
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
CMD ["python3", "/home/JetsonYolo/JetsonYolo.py"]