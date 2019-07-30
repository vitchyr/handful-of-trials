FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install pip
RUN apt-get update
RUN apt-get -y install python3 python3-pip python3-dev python3-tk
RUN apt-get -y install libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev

# Install basic libraries
RUN pip3 install --upgrade pip
RUN pip3 install numpy tensorflow-gpu==1.9 matplotlib scipy scikit-learn future

# Install MuJoCo + OpenAI gym
RUN pip3 install gym==0.9.4
RUN apt-get update
RUN apt-get -y install unzip unetbootin wget
RUN mkdir -p /.mujoco && cd /.mujoco && wget https://www.roboti.us/download/mjpro131_linux.zip && unzip mjpro131_linux.zip
ENV MUJOCO_PY_MJKEY_PATH="/root/.mujoco/mjkey.txt"
ENV MUJOCO_PY_MJPRO_PATH="/root/.mujoco/mjpro131"
RUN pip3 install mujoco-py==0.5.7

# Install additional requirements
RUN pip3 install datetime gitpython h5py tqdm dotmap cython

# GPFlow
RUN apt-get -y install git
RUN git clone https://github.com/GPflow/GPflow.git
RUN pip3 install pandas multipledispatch pytest
RUN cd GPflow/ && pip install . --no-deps

# Create copy of Deep MBRL repo and place in ~/handful-of-trials
RUN cd ~ && git clone https://github.com/kchua/handful-of-trials.git

# Environment setup
RUN echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu' >> /root/.bashrc
RUN echo 'alias python=python3' >> /root/.bashrc

RUN pip3 install cloudpickle==0.5.2
RUN pip3 install tensorflow==1.14
RUN pip3 install decorator==4.4.0
RUN pip3 install pygame==1.9.6
RUN pip3 install ipdb==0.12
RUN apt-get -y install ffmpeg
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN ln -s /root/.mujoco/mujoco200_linux/ /root/.mujoco/mujoco200
COPY ./mjkey.txt /root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}
RUN apt-get -y install libosmesa6-dev curl
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf
RUN pip3 install mujoco-py==2.0.2.2
RUN pip3 install gym==0.12.5

CMD /bin/bash
