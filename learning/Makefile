SHELL := /bin/bash
VIRTUALENV = LF_IL_virtualenv_learning
AG_VERSION = 2019-04-16

# Makefile to structure work on the AI-DO baselines.

# ---------- DOCKER --------------
# THIS WILL ONLY RUN ON LINUX, NOT on Mac, NOT on Windows since nvidia-docker is not supported
# Nvidia-docker is covered here: https://github.com/NVIDIA/nvidia-docker
# and https://devblogs.nvidia.com/gpu-containers-runtime/
prepare-docker:
				sudo apt-get update; \
				docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f; \
				sudo apt-get purge -y nvidia-docker; \
				curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -; \
				curl -s -L https://nvidia.github.io/nvidia-docker/$$(. /etc/os-release;echo $$ID$$VERSION_ID)/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list; \
				sudo apt-get update; \
				sudo apt-get install -y nvidia-docker2 nvidia-container-runtime; \
				sudo pkill -SIGHUP dockerd; \
				docker run --runtime=nvidia --rm nvidia/cuda:10.0-base nvidia-smi;

# This may run for >10 min and longer (on GPU) if more data is used for learning.
learn-docker:
				docker build -t learn_container .; \
				mkdir learned_models; \
				docker run --runtime=nvidia -v $$PWD/learned_models:/workspace/learned_models learn_container;

# Extract data from docker container and copies it to the learning repository
copy_for_submission:
				sudo chown $$(id -u):$$(id -g) learned_models/frozen_graph.pb; \
				cp learned_models/frozen_graph.pb ../imitation_agent/;

# ---------- REGULAR --------------
# This assumes that you have GPU drivers installed already.

# If only CPU is available for computation on your machine,
# change the line "pip install tensorflow-gpu" to "tensorflow" (without the GPU option)
install-dependencies:
				virtualenv $(VIRTUALENV) --python=python3.7; \
				. $(VIRTUALENV)/bin/activate; \
				pip install --no-cache-dir -r requirements.txt; \
				pip install --upgrade tensorflow-gpu==1.13.1;

learn-regular:
				. $(VIRTUALENV)/bin/activate; \
				python src/cnn_training_tensorflow.py; \
				python src/freeze_graph.py;

regular_copy_for_submission:
				cp learned_models/frozen_graph.pb ../imitation_agent;
