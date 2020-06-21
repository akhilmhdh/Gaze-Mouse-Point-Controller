import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
import cv2

class FaceDetectionModel:
    '''
    Class for the Face Detection Model.
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

    def predict(self, image,threshold):
        input_img = self.preprocess_input(image)
        input_dict = {self.input_name:input_img}
        outputs = self.exec_network.infer(input_dict)[self.output_name]
        coords = self.preprocess_outputs(outputs,threshold,(image.shape[1],image.shape[0]))
        return self.crop_face(coords,image)

    def crop_face(self, coords, image):
        # TODO: This method needs to be completed by you
        if(len(coords)==1):
            coords = coords[0]
            cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]
            return coords,cropped_face
        return False,False

    def preprocess_input(self, image):
        preprocessed_frame = cv2.resize(image,(self.input_shape[3],self.input_shape[2]))
        preprocessed_frame = preprocessed_frame.transpose((2,0,1))
        return preprocessed_frame.reshape(1,*preprocessed_frame.shape)

    def preprocess_outputs(self,outputs,threshold,dim):
        li=[]
        for box in outputs[0][0]:
            ct = box[2]
            if ct > threshold:
                xmin=int(box[3]*dim[0])
                ymin=int(box[4]*dim[1])
                xmax=int(box[5]*dim[0])
                ymax=int(box[6]*dim[1])
                li.append([xmin,ymin,xmax,ymax])
        return li
