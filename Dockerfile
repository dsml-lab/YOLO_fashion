FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04
RUN apt-get update && apt-get install -y \
	sudo \
	wget \
	vim \
	git \
	zip
WORKDIR /opt
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh && \
	sh /opt/Anaconda3-2021.05-Linux-x86_64.sh -b -p /opt/anaconda3 && \
	rm -f /opt/Anaconda3-2021.05-Linux-x86_64.sh
ENV PATH /opt/anaconda3/bin:$PATH

RUN pip install --upgrade pip

RUN conda install \
	pytorch \
	torchvision \
	cudatoolkit=11.1 -c pytorch-lts -c nvidia
RUN apt-get install -y libgl1-mesa-dev
	
WORKDIR /workspace
COPY requirements.txt /workspace
RUN pip install -r requirements.txt

CMD ["bash"]
