import os
import sys
import logging as log
from openvino.inference_engine import IECore
import cv2
import math
class GazeEstimationModel:
    '''
    Class for the Gaze Est Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
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
                    msg = "Layer extension doesn't support all layers"
                    log.error(msg)
                    raise Exception(msg)
        
        self.exec_network= self.plugin.load_network(self.network, self.device)

        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_name=next(iter(self.network.outputs))
        self.output_shape=self.network.outputs[self.output_name].shape

        return

    def predict(self,left_eye,right_eye,head_pose,preview=None):
        input_left_eye,input_right_eye = self.preprocess_input(left_eye,right_eye)
        input_dict = {'left_eye_image':input_left_eye,'right_eye_image':input_right_eye,'head_pose_angles':head_pose}
        outputs = self.exec_network.infer(input_dict)[self.output_name]
        if preview is not None:
            return self.preprocess_outputs(outputs[0],head_pose[2],preview)
        return self.preprocess_outputs(outputs[0],head_pose[2])

    def preprocess_input(self, left_eye,right_eye):
        preprocessed_left_eye = cv2.resize(left_eye,(self.input_shape[3],self.input_shape[2]))
        preprocessed_left_eye = preprocessed_left_eye.transpose((2,0,1))

        preprocessed_right_eye = cv2.resize(right_eye,(self.input_shape[3],self.input_shape[2]))
        preprocessed_right_eye = preprocessed_right_eye.transpose((2,0,1))

        return preprocessed_left_eye.reshape(1,*preprocessed_left_eye.shape),preprocessed_right_eye.reshape(1,*preprocessed_right_eye.shape)

    def preprocess_outputs(self,outputs,roll,preview=None):
        cos = math.cos(roll*math.pi/180.0)
        sin = math.sin(roll*math.pi/180.0)

        x = outputs[0] * cos + outputs[1] * sin
        y = outputs[0] * sin + outputs[1] * cos 

        if preview is not None:
            cv2.putText(preview, "Gaze Coord: x:{:.2f}|| y:{:.2f} || z:{:.2f}".format(outputs[0],outputs[1],outputs[2]), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
            return (x,y),outputs,preview
        return (x,y),outputs
        
