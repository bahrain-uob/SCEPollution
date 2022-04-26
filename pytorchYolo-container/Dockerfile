FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3
RUN pip3 install -U pip setuptools
RUN pip3 install pandas
RUN pip3 install requests
RUN pip3 install tqdm 
RUN pip3 install PyYaml
RUN pip3 install matplotlib
RUN pip3 install opencv-python
RUN pip3 install scipy
RUN pip3 install scikit-learn==0.21.3
CMD ["python3", "/home/JetsonYolo/JetsonYolo.py"]

