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
                        help="single image mode yes/no", default="cam")
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
    parser.add_argument("-flags", "--preview_flags", required=False, nargs='+',
                        default=[],
                        help="flags to set intermediate flags.Space between each. fl: facial landmark || hp: head pose || ge: gaze estimation")
        
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def infer_on_stream(args):

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    intermediatePreview = args.preview_flags

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

    log.info("Model loading...")
    # model loading
    model_loading = time.time()

    # inference pipeline
    face_detector.load_model()
    face_landmark_detector.load_model()
    head_pose_estimation.load_model()
    gaze_estimation.load_model()

    log.info("Models are loaded")
    log.info("Modal Loading Time: {:.3f}ms".format((time.time() - model_loading)* 1000))

    # visual pipeline
    try:
        input_feeder = InputFeeder(input_type,input_file)
        input_feeder.load_data()
    except:
        log.error("Something went wrong with loading camera/mouse")
        exit(0)

    mouse = MouseController(precision,speed)
    frames = 0

    for ret,frame in input_feeder.next_batch():
        if not ret:
            break
        frames+=1

        key = cv2.waitKey(60)

        inf_start = time.time()

        face_coords,face_cropped_image=face_detector.predict(frame,prob_threshold)
        preview_image = face_cropped_image

        if (face_coords):
            if 'fl' in intermediatePreview:
                eye_coords, left_eye,right_eye, preview_image = face_landmark_detector.predict(face_cropped_image,True)
            else:
                eye_coords, left_eye,right_eye, preview_image = face_landmark_detector.predict(face_cropped_image)
            
            if 'hp' in intermediatePreview:
                head_pose_angles,preview_image = head_pose_estimation.predict(face_cropped_image,preview_image)
            else:
                head_pose_angles = head_pose_estimation.predict(face_cropped_image)

            if 'ge' in intermediatePreview:
                mouse_coord,gaze_coord,preview_image = gaze_estimation.predict(left_eye,right_eye,head_pose_angles,preview_image)
            else:
                mouse_coord,gaze_coord = gaze_estimation.predict(left_eye,right_eye,head_pose_angles)

            left_eye = (eye_coords[0][0]+20,eye_coords[0][1]+20)
            right_eye = (eye_coords[1][0]+20,eye_coords[1][1]+20)

            gaze_x = int(gaze_coord[0] * 250)
            gaze_y = int(-gaze_coord[1] * 250)

            if 'ge' in intermediatePreview:
                cv2.arrowedLine(preview_image, left_eye, (left_eye[0]+gaze_x,left_eye[1]+gaze_y),(0, 255, 0), 3)
                cv2.arrowedLine(preview_image, right_eye, (right_eye[0]+gaze_x,right_eye[1]+gaze_y),(0, 255, 0), 3)
            
            inference_time = time.time() - inf_start

            inf_time_message = "Inf Time Per Frame: {:.3f}ms"\
                               .format(inference_time * 1000)

            cv2.putText(preview_image,inf_time_message, (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
        
        cv2.imshow('frame',cv2.resize(preview_image,(400,400)))

        if frames % 5 == 0:
            mouse.move(mouse_coord[0],mouse_coord[1])

    input_feeder.close()



def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    log.basicConfig(level=log.DEBUG)
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    infer_on_stream(args)


if __name__ == '__main__':
    main()
