FROM tensorflow/tensorflow:latest-gpu-py3

# let's create our workspace, we don't want to clutter the container
RUN rm -r /workspace; mkdir /workspace

WORKDIR /workspace

# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir tensorflow-gpu==1.13.1

# necessary to get opencv to work properly
RUN apt update && apt install -y libsm6 libxext6

# Train CNN model for lane following && Freeze model graph
CMD python src/cnn_training_tensorflow.py && python src/freeze_graph.py
