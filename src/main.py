import os , cv2
from src.util_logger import setup_logger_linux
logger = setup_logger_linux(module_name=str(__name__))

# from src.read_cam.read_webcam import CV2VideoCapture
# from src.analysis.detr_hugging_face import (GetFramesFromVids , 
#                                         PlotBboxOnFrames,
#                                         FaceDetection,
#                                         ObjDetHFRtDetr) #,PlotBboxOnFrames

# from src.analysis.hugging_face_rtdetr_v2 import AutoModelRtDetrV2
# from src.analysis.gender_detect import DeepfaceDetect #XceptionFaceClass
# from src.analysis.ultralytics_yoloe import UltraLyticsYoloeYe ## HOLD 
# from src.analysis.pose_google_media_pipe import MediaPipeGoog
# from src.analysis.ultraLytics_fastSAM_1 import FastSAMProcessor
#print(f"-main-hit-> ")

class IPWebCam:
    """ 
    """

    @classmethod
    def get_frames_local_list(self,root_dir):
        """ 
        """
        try:
            #root_dir = "../data_dir/out_vid_frames_dir/" #root_dir = "static/image_uploads/"
            ls_files_uploads = []
            for root, dirs, files in os.walk(root_dir):
                for filename in files:
                    ls_files_uploads.append(os.path.join(root, filename))
            logger.debug("-List of Image Files Uploaded ----> %s",ls_files_uploads)
            return ls_files_uploads
        except Exception as err:
            logger.error("-Error--get_frames_local_list---> %s" ,err)

    # @classmethod
    # def invoke_scan(self):
    #     """`
    #     """
    #     print(f"-invoke_scan--hit-> ")
    #     logger.debug(f"-invoke_scan--hit-> ")
    #     CV2VideoCapture().video_cap_init()

    # @classmethod
    # def analyse_scan(self):
    #     """`
    #     """
    #     print(f"-analyse_scan--hit-> ")

    #     GetFramesFromVids().get_frame_from_video()
    #     PlotBboxOnFrames().get_bbox_on_frames()

    @classmethod
    def face_detect_yolo_hface(self):
        """ 
        """
        image_rootDIR= "../data_dir/jungle_images/input_DIR/"
        ls_files_uploads = self.get_frames_local_list(image_rootDIR)
        for iter_k in range(len(ls_files_uploads)):
            image_local_path = ls_files_uploads[iter_k]
            print("--image_local_path----",image_local_path)
            rectangle_portion_only = cv2.imread(image_local_path)
            #FaceDetection().get_colors_from_face(rectangle_portion_only,face_bbox_coords_detection=None)
            FaceDetection().face_detect_yolo_huggin_face(image_local_path)

    # @classmethod
    # def xception_gender_class(self):
    #     """ 
    #     """
    #     image_rootDIR= "../data_dir/jungle_images/input_DIR/"
    #     ls_files_uploads = self.get_frames_local_list(image_rootDIR)
    #     for iter_k in range(len(ls_files_uploads)):
    #         image_local_path = ls_files_uploads[iter_k]
    #         print("--image_local_path----",image_local_path)
    #         XceptionFaceClass().detect_gender(image_local_path)

    @classmethod
    def deepface_detect_gender(self):
        """ 
        """
        ls_dominant_race = []
        ls_dominant_gender = []
        image_rootDIR= "../data_dir/jungle_images/input_DIR/"
        ls_files_uploads = self.get_frames_local_list(image_rootDIR)
        for iter_k in range(len(ls_files_uploads)):
            image_local_path = ls_files_uploads[iter_k]
            #print("--image_local_path----",image_local_path)
            DeepfaceDetect().extract_faces(image_local_path)
            #DeepfaceDetect().get_gender_obj(image_local_path)


    @classmethod
    def object_detect_HFRtDetr_pipeline(self):
        """ 
        Desc:
            - pipeline processed - Not direct Model 
        """
        try:
            image_frame_path = "../data_dir/jungle_images/input_DIR/"
            ls_files_uploads = self.get_frames_local_list(image_frame_path)
            for iter_k in range(len(ls_files_uploads)):
                image_frame = ls_files_uploads[iter_k]
                print("--IMAGE--FRAME-----",image_frame)
                print("   ==FRA------   "*20)
                ObjDetHFRtDetr().object_detect_RT_DETR(image_frame)
        except Exception as err:
            print(err)

    @classmethod
    def object_detect_HFRtDetr_model(self):
        """ 
        Desc:
            - pipeline processed - Not direct Model 
        """
        try:
            image_rootDIR= "../data_dir/jungle_images/input_DIR/"
            ls_files_uploads = self.get_frames_local_list(image_rootDIR)
            for iter_k in range(len(ls_files_uploads)):
                image_local_path = ls_files_uploads[iter_k]
                print("--image_local_path----",image_local_path)
                image_detections , image_local_frame = AutoModelRtDetrV2().obj_detect_HFRtDetr_v2_model(image_local_path)
                logger.debug("--main.py--model_obj_detection--image_detections----aa---> %s" ,image_detections)
                AutoModelRtDetrV2().plot_results_HFRtDetr_v2_model(image_detections , image_local_frame,image_local_path)
        except Exception as err:
            logger.error("--main.py--object_detect_HFRtDetr_model-> %s" ,err)

    @classmethod
    def get_multi_cam_alert(self):
        """ 
        """
        try:
            CV2VideoCapture().video_cap_multi_cam()
            ## FaceDetection().face_detect_yolo_huggin_face(image_local_path)
            
        except Exception as err:
            logger.error("--main.py--get_multi_cam_alert---> %s" ,err)


    # @classmethod
    # def ultralytics_yoloe_ye(self):
    #     """ 
    #     # TODO -- Hold maybe Not required -- 
    #     # Directly get their model from the HuggingFace Hub in earlier own code flow 
    #     """
    #     try:
    #         ## FaceDetection().face_detect_yolo_huggin_face(image_local_path)

    #         UltraLyticsYoloeYe().test_1()
            
            
    #     except Exception as err:
    #         logger.error("--main.py--ultralytics_yoloe_ye---> %s" ,err)

    
    @classmethod
    def pose_media_pipe_google(self):
        """ 
        Desc:
            - Not IPWebCam -- Pose detection 
            - Hit the recorded Videos and Static Frames 

        """
        try:
            print("--HIT--pose_media_pipe_google---")

            MediaPipeGoog().pose_media_pipe_google_2()
            
        except Exception as err:
            logger.error("--main.py--pose_media_pipe_google---> %s" ,err)


    @classmethod
    def ultralytics_fast_sam(self):
        """ 
        Desc:
            - Not IPWebCam -- Segment Anything - fastSAM
            - Hit the recorded Videos and Static Frames 

        """
        try:
            print("--HIT--ultralytics_fast_sam---")
            FastSAMProcessor().process_images_from_directory()
        except Exception as err:
            logger.error("--main.py--ultralytics_fast_sam--> %s" ,err)



if __name__ == "__main__":
    #IPWebCam().invoke_scan() #TODO -ARGPARSE required for main method calls
    #IPWebCam().analyse_scan()
    #IPWebCam().face_detect_yolo_hface() #  #TODO -- OK -
    #IPWebCam().get_multi_cam_alert() # #TODO -- HOLD faces -- testing with YOLOE 
    #IPWebCam().pose_media_pipe_google() ## OK 
    ## PATH -- cd /home/dhankar/temp/01_25/git_up_ipWebCam/ipWebCam/data_dir/pose_detected/init_video
    
    #IPWebCam().ultralytics_yoloe_ye() ## #TODO -- HOLD laterz 
    #IPWebCam().xception_gender_class()
    #IPWebCam().deepface_detect_gender() #TODO -- OK -- get input from -- face_detect_yolo_hface
    #IPWebCam().object_detect_HFRtDetr_pipeline()
    IPWebCam().object_detect_HFRtDetr_model()
    #IPWebCam().ultralytics_fast_sam()
  

# import argparse
# parser = argparse.ArgumentParser(description='overlander_main_args')
# parser.add_argument('--root_dir', help='root_dir_overlander', nargs='?', const=0)
# parser.add_argument('--perc_split', help='train_test_split', nargs='?', const=0)
# parser.add_argument('--data_type_flag', help='dataType', nargs='?', const=0)
