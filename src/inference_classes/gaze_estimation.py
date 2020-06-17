import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Model_X:
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
        self.model=model_name


    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        model_xml = self.model
        model_bin = os.path.splitext(self.model)[0]+".bin"
        self.plugin = IECore()
        # network model
        self.network = IENetwork(model=model_xml, weights=model_bin)

        network_layers = self.network.layers.keys()
        supported_layers = self.plugin.query_network(network=self.network,device_name=device).keys()
        ext_required = False

        for layer in network_layers:
            if layer in supported_layers:
                pass
            else:
                ext_required= True
                break
        
        # cpu extension added
        if self.extensions and "CPU" in device and ext_required:
            self.plugin.add_extension(cpu_extension, device)
        
        self.exec_network= self.plugin.load_network(self.network, self.device)

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

        return

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_img = self.preprocess_input(image)
        input_dict = {self.input_name:input_img}
        outputs = self.net.infer(input_dict)[self.output_name]
        coords = self.preprocess_outputs(outputs,(image.shape[1],image.shape[0]))
        return self.draw_outputs(coords,image)

    def draw_outputs(self, coords, image):
        # TODO: This method needs to be completed by you
        for coord in coords:
            cv2.rectangle(image, (coord[0],coord[1]), (coord[2], coord[3]), (0, 55, 255), 1)
        return coords,image

    def preprocess_input(self, image):
          '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        preprocessed_frame = cv2.resize(image,(self.input_shape[3],self.input_shape[2]))
        preprocessed_frame = preprocessed_frame.transpose((2,0,1))
        return preprocessed_frame.reshape(1,*preprocessed_frame.shape)

    def preprocess_outputs(self, outputs,dim):
        # TODO: This method needs to be completed by you
        li=[]
        for box in outputs[0][0]:
            ct = box[2]
            if ct > self.threshold:
                xmin=int(box[3]*dim[0])
                ymin=int(box[4]*dim[1])
                xmax=int(box[5]*dim[0])
                ymax=int(box[6]*dim[1])
                li.append([xmin,ymin,xmax,ymax])
        return li
