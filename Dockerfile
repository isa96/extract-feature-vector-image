FROM ubuntu:20.04


#FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# FROM nvidia/cuda:11.0.3-base-ubuntu20.04

RUN apt-get update 
RUN apt-get upgrade -y

RUN echo "Installing OS (apt) dependencies..."
ENV TZ="Asia/Jakarta"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#Kebutuhan Visual Mode 1 (HTTP)
RUN apt-get install -y libgtk-3-0
RUN apt-get install -y libgstreamer1.0-0
RUN apt-get install -y libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN apt-get install -y libgtk2.0-dev

#Kebutuhan Visual Mode 2 (RTSP)
RUN apt-get  install -y libcairo2-dev libgirepository1.0-dev
RUN apt-get  install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0
RUN apt-get  install -y libglib2.0-dev libgstrtspserver-1.0-dev gstreamer1.0-rtsp
RUN apt-get  install -y gir1.2-gst-rtsp-server-1.0
#RUN apt-get  install -y nano
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y xauth

RUN apt-get install -y python3-pip

WORKDIR /app 

COPY . .

RUN pip3 install torch torchvision opencv-python

RUN apt-get clean && rm -rf /var/lib/apt/lists/* && rm -rf *.whl && rm -rf *.txt
