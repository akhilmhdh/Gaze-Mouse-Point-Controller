import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
import cv2

class HeadPoseEstimationModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        self.infer_req = None
        self.device = device
        self.extensions=extensions
        self.model_path=model_name

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        model_xml = self.model_path
        model_bin = os.path.splitext(self.model_path)[0]+".bin"
        self.plugin = IECore()
        # network model
        self.network = self.plugin.read_network(model=model_xml, weights=model_bin)

        network_layers = self.network.layers.keys()
        supported_layers = self.plugin.query_network(network=self.network,device_name=self.device).keys()
        ext_required = False

        for layer in network_layers:
            if layer in supported_layers:
                pass
            else:
                ext_required= True
                break
        
        # cpu extension added
        if self.extensions!=None and "CPU" in self.device and ext_required:
            self.plugin.add_extension(self.extensions, self.device)

            for layer in network_layers:
                if layer in supported_layers:
                    pass
                else:
                    raise Exception("Layer extension doesn't support all layers")
        
        self.exec_network= self.plugin.load_network(self.network, self.device)

        self.input_name=next(iter(self.network.inputs))
        self.input_shape=self.network.inputs[self.input_name].shape
        self.output_name=next(iter(self.network.outputs))
        self.output_shape=self.network.outputs[self.output_name].shape

        return

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_img = self.preprocess_input(image)
        input_dict = {self.input_name:input_img}
        outputs = self.exec_network.infer(input_dict)
        angles = self.preprocess_outputs(outputs,(image.shape[1],image.shape[0]))
        return angles

    def preprocess_input(self, image):
        preprocessed_frame = cv2.resize(image,(self.input_shape[3],self.input_shape[2]))
        preprocessed_frame = preprocessed_frame.transpose((2,0,1))
        return preprocessed_frame.reshape(1,*preprocessed_frame.shape)

    def preprocess_outputs(self,outputs,dim):
        # TODO: This method needs to be completed by you
        li=[]
        for key in outputs:
            li.append(outputs[key][0][0])
        return li
