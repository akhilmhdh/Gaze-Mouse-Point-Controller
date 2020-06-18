"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


# MQTT server environment variables
import os
import sys
import time
import cv2
import numpy as np
import logging as log
from argparse import ArgumentParser

from inference_classes.face_detection import FaceDetectionModel
from inference_classes.facial_landmarks_detection import FacialLandmardDetectionModel
from inference_classes.head_pose_estimation import HeadPoseEstimationModel
from inference_classes.gaze_estimation import GazeEstimationModel

from input_feeder import InputFeeder
from mouse_controller import MouseController

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd_m", "--face_detector_model", required=True, type=str,
                        help="Path to a face detection xml file with a trained model.")
    parser.add_argument("-hp_m", "--head_pose_estimation_model", required=True, type=str,
                        help="Path to a head pose estimation xml file with a trained model.")
    parser.add_argument("-fld_m", "--facial_landmark_model", required=True, type=str,
                        help="Path to a facial landmark detection xml file with a trained model.")
    parser.add_argument("-ge_m", "--gaze_estimation_model", required=True, type=str,
                        help="Path to a gaze estimation xml file with a trained model.")                    
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-mc_prec", "--mouse_prec", required=False, type=str,default="medium",
                        help="mouse precision needed values are high-low-medium")
    parser.add_argument("-mc_speed", "--mouse_speed", required=False, type=str,default="fast",
                        help="mouse speed needed values are fast-slow-medium")
    parser.add_argument("-type", "--type", required=False, type=str,
                        help="single image mode yes/no", default="video")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def infer_on_stream(args):

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    face_detector_path = args.face_detector_model
    facial_landmark_path = args.facial_landmark_model
    head_pose_path = args.head_pose_estimation_model
    gaze_est_path = args.gaze_estimation_model

    device = args.device
    extension = args.cpu_extension
    input_type = args.type.lower()
    input_file = args.input

    speed = args.mouse_speed
    precision = args.mouse_prec

    # model classess intializing
    face_detector = FaceDetectionModel(model_name=face_detector_path,device=device,extensions=extension)
    face_landmark_detector = FacialLandmardDetectionModel(model_name=facial_landmark_path,device=device,extensions=extension)
    head_pose_estimation = HeadPoseEstimationModel(model_name=head_pose_path,device=device,extensions=extension)
    gaze_estimation = GazeEstimationModel(model_name=gaze_est_path,device=device,extensions=extension)

    # model loading
    face_detector.load_model()
    face_landmark_detector.load_model()
    head_pose_estimation.load_model()
    gaze_estimation.load_model()

    # visual pipeline
    input_feeder = InputFeeder(input_type,input_file)
    input_feeder.load_data()

    mouse = MouseController(precision,speed)
    frames = 0

    for ret,frame in input_feeder.next_batch():
        if not ret:
            break
        frames+=1

        key = cv2.waitKey(60)

        face_coords,face_cropped_image=face_detector.predict(frame,prob_threshold)
        eye_coords, left_eye,right_eye = face_landmark_detector.predict(face_cropped_image)
        head_pose_angles = head_pose_estimation.predict(face_cropped_image)
        mouse_coord,gaze_coord = gaze_estimation.predict(left_eye,right_eye,head_pose_angles)
        
        cv2.imshow('frame',face_cropped_image)

        if frames % 3 == 0:
            mouse.move(mouse_coord[0],mouse_coord[1])

    input_feeder.close()



def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)


if __name__ == '__main__':
    main()
