import os
import sys
import logging as log
from openvino.inference_engine import IECore
import cv2

class FacialLandmardDetectionModel:
    '''
    Class for the Face LandMark detection Model.
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

        self.input_name=next(iter(self.network.inputs))
        self.input_shape=self.network.inputs[self.input_name].shape
        self.output_name=next(iter(self.network.outputs))
        self.output_shape=self.network.outputs[self.output_name].shape

        return

    def predict(self, image,preview=False):
        input_img = self.preprocess_input(image)
        input_dict = {self.input_name:input_img}
        outputs = self.exec_network.infer(input_dict)[self.output_name]
        coords = self.preprocess_outputs(outputs,(image.shape[1],image.shape[0]))
        return self.draw_outputs(coords,image,preview)

    def draw_outputs(self, coords, image, preview):
        left_eye_min = (coords[0]-15,coords[1]-15)
        left_eye_max = (coords[0]+15,coords[1]+15)
        right_eye_min = (coords[2]-15,coords[3]-15)
        right_eye_max = (coords[2]+15,coords[3]+15)
        left_eye = image[left_eye_min[1]:left_eye_max[1],left_eye_min[0]:left_eye_max[0]]
        right_eye = image[right_eye_min[1]:right_eye_max[1],right_eye_min[0]:right_eye_max[0]]

        if preview:
            cv2.rectangle(image,left_eye_min,left_eye_max,(200, 10, 10),2)
            cv2.rectangle(image,right_eye_min,right_eye_max,(200, 10, 10),2)
        
        eye_coords = [[left_eye_min[0],left_eye_min[1],left_eye_max[1],left_eye_max[1]],
        [right_eye_min[0],right_eye_min[1],right_eye_max[0],right_eye_max[1]]]
        return eye_coords,left_eye,right_eye,image

    def preprocess_input(self, image):
        preprocessed_frame = cv2.resize(image,(self.input_shape[3],self.input_shape[2]))
        preprocessed_frame = preprocessed_frame.transpose((2,0,1))
        return preprocessed_frame.reshape(1,*preprocessed_frame.shape)

    def preprocess_outputs(self,outputs,dim):
        left_eye_x=int(outputs[0][0]*dim[0])
        left_eye_y=int(outputs[0][1]*dim[1])
        right_eye_x=int(outputs[0][2]*dim[0])
        right_eye_y=int(outputs[0][3]*dim[1])

        return (left_eye_x,left_eye_y,right_eye_x,right_eye_y)
