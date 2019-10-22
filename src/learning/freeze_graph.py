#!/usr/bin/env python

import os
from tensorflow.python.tools import freeze_graph

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # Define the name of your model
    model_name = 'learned_models'

    # define the path to the graph from training
    input_graph = os.path.join(os.getcwd(), model_name, 'tensorflow_logs', 'graph', 'graph.pb')

    # define the path in which to save the frozen graph
    output_graph = os.path.join(os.getcwd(), model_name, 'frozen_graph.pb')

    # the frozen_graph directory must exist in order to freeze the model
    directory = os.path.join(os.getcwd(), model_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # define the checkpoint/weights you want to freeze inside the graph
    input_checkpoint = os.path.join(os.getcwd(), model_name, 'tensorflow_logs', 'train-900')

    # define the name of the prediction output node
    # This name can be easily extracted using Tensorboard. In GRAPHS tab of Tensorboard, check the inputs of Loss scope.
    # In this case they are "vel_true" and "ConvNet/fc_layer_2/BiasAdd".The CNN's predictions are provided from the
    # "ConvNet/fc_layer_2/BiasAdd" element, whereas the true omega velocities from the "vel_true". Here we have to define
    # the element which provides the CNN's predictions and thus we defined as output_node_names the "ConvNet/fc_layer_2/BiasAdd".
    output_node_names = "ConvNet/fc_layer_2/BiasAdd"

    # The following settings should remain the same
    input_saver = ""
    input_binary = True
    restore_op_name = 'save/restore_all'
    filename_tensor_name = 'save/Const:0'
    clear_devices = True
    initializer_nodes = ""
    variable_names_blacklist = ""

    # Freeze the graph
    freeze_graph.freeze_graph(
        input_graph,
        input_saver,
        input_binary,
        input_checkpoint,
        output_node_names,
        restore_op_name,
        filename_tensor_name,
        output_graph,
        clear_devices,
        initializer_nodes,
        variable_names_blacklist
    )
    print("The frozen graph is saved in {}.".format(output_graph))


if __name__ == '__main__':
    main()
