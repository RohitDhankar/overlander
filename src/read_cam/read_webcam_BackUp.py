import numpy as np
from util_logger import setup_logger_linux
logger = setup_logger_linux(module_name=str(__name__))
import cv2 , time , os 
from datetime import datetime
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

from PIL import Image, ImageDraw
from analysis.detr_hugging_face import (GetFramesFromVids , 
                                        PlotBboxOnFrames,
                                        FaceDetection,
                                        ObjDetHFRtDetr) #,PlotBboxOnFrames


class CV2VideoCapture:
    """
    Desc:
        - CV2VideoCapture 
    """

    @classmethod
    def get_init_dir(cls):
        """ 
        """
        try:
            minute_now = datetime.today().strftime('%Y-%m-%d-%H_%M')
            hour_now = datetime.today().strftime('%Y-%m-%d-%H_%M')
            #init_vid_dir = '../data_dir/init_vid_dir_' + str(minute_now) + "_/" #initial vid dir 
            init_vid_dir = '../data_dir/init_vid_dir_' + str(hour_now) + "_/" 
            if not os.path.exists(init_vid_dir):
                os.makedirs(init_vid_dir)
            return str(init_vid_dir)
        except Exception as err:
            logger.error("-Error--get_init_dir---> %s" , err)

    @classmethod
    def video_cap_init(cls):
        """
        """
        try:
            print(f"-video_cap_init---aa---hit-> ")
            logger.debug(f"-video_cap_init--hit-> ")
            init_vid_dir = cls.get_init_dir()
            #cap_local_cam = cv2.VideoCapture(0) # local_laptop_camera # cap = cv2.VideoCapture('rtsp://[name]:[password]@192.168.1.32/stream1')
            cap_tapo = cv2.VideoCapture('rtsp://tapo12345:SomePass@1234@192.168.1.42/stream1') #TODO - Port Changes TAPO video stream
            #cap_ezviz = cv2.VideoCapture('rtsp://admin:IJKGIP@192.168.1.41:8000') # EZVIZ -video stream - ##554/h264_stream
            print("--ok-VideoCapture--->",type(cap_tapo)) #<class 'cv2.VideoCapture'>
            logger.debug("-upload_image--aa-> %s" , type(cap_tapo)) #<class 'cv2.VideoCapture'> #cv2.imshow(cap_tapo)
            w = cap_tapo.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap_tapo.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap_tapo.get(cv2.CAP_PROP_FPS) 
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            counter_video_capture = 0
            while True:
                if counter_video_capture >=2:
                    return "done-3-vids"
                else:
                    date_time_now  = datetime.now().isoformat().replace(':','-').replace('.','-')
                    vid_file_name = "video_file__" + str(date_time_now) + '_.mp4'
                    output_stream = cv2.VideoWriter(init_vid_dir+vid_file_name, 
                                                    fourcc, 
                                                    fps, 
                                                    (int(w),int(h)))
                    start_time = time.time() # start timer
                    while (int(time.time() - start_time) < 10):
                        ret, frame = cap_tapo.read() ## Capture vid per 20 sec -- #print("--Type---",type(frame)) ##<class 'numpy.ndarray'>
                        if ret==True:
                            #cv2.imshow('frame', frame) ## OK -- Dont 
                            output_stream.write(frame)
                    logger.debug("-Written--Video----> %s" , vid_file_name)
                    print("-Written--Video-",counter_video_capture)
                    counter_video_capture += 1
                    output_stream.release()
        except Exception as err:
            logger.error("-Error--video_cap_init-> %s" , err)



    @classmethod
    def invoke_model_yolov8_face_detection(cls):
        """ 
        """
        # download model
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        print("----model_path--> %s" ,model_path)
        model_yolov8_face_detection = YOLO(model_path) # load model
        print("---model_yolov8_face_detection--> %s" ,type(model_yolov8_face_detection))
        return model_yolov8_face_detection
    
    @classmethod
    def vid_frame_face_detect_yolo_huggin_face(cls,resized_frame1):
        """ 
        """
        
        model_yolov8_face_detection = cls.invoke_model_yolov8_face_detection()
        #image_for_process = cv2.imread(image_frame_path) ## image_for_process = cv2.imdecode(Image.open(image_frame_path))
        #image_for_roi = Image.open(image_frame_path)
        image_for_roi = resized_frame1

        image_local_path= "../data_dir/jungle_images/input_DIR/hostes_2.png"
        # ls_files_uploads = self.get_frames_local_list(image_rootDIR)
        # for iter_k in range(len(ls_files_uploads)):
        # image_local_path = ls_files_uploads[iter_k]
        # print("--image_local_path----",image_local_path)
        rectangle_portion_only = cv2.imread(image_local_path)
        print("----rectangle_portion_only----->>",type(rectangle_portion_only))

        # output = model_yolov8_face_detection(Image.open(rectangle_portion_only))
        # print("---FACE-----output-----RESULTS",output)
        # results_face_detect = Detections.from_ultralytics(output[0]) #detections = sv.Detections(...)
        # print("---FACE---RESULTS",results_face_detect)




    @classmethod
    def video_cap_multi_cam(cls):
        """
        """
        import time

        # Frame counter and save interval (in seconds)
        image_saved_last = ""
        frame_counter = 0
        save_interval = 2  # Save a frame every 3 seconds
        last_save_time = time.time()

        cap_tapo_1 = cv2.VideoCapture('rtsp://tapo12345:SomePass@1234@192.168.1.32/stream1') #TODO 
        # cap_tapo_2 = cv2.VideoCapture('rtsp://tapo12345:SomePass@1234@192.168.1.32/stream1') #TODO 
        # cap_tapo_3 = cv2.VideoCapture('rtsp://tapo12345:SomePass@1234@192.168.1.32/stream1') #TODO 
        # cap_tapo_4 = cv2.VideoCapture('rtsp://tapo12345:SomePass@1234@192.168.1.32/stream1') #TODO 

        # Check if all cameras are opened correctly
        # if not (cap_tapo_1.isOpened() and cap_tapo_2.isOpened() and cap_tapo_3.isOpened() and cap_tapo_4.isOpened()):
        #     print("Cannot open one or more cameras")
        #     #exit() # TODO -- which camera failed #VMS_reqmt
        # #
        
        image_root_dir= "../data_dir/face_detected/frame_for_face/"
        
        while True:
            # Capture frame-by-frame from each camera
            ret1, frame1 = cap_tapo_1.read()
            print('Size of Image:', frame1.size) ## Pixels --- Size of Image:
            # Resize frames to the desired size
            resized_frame1 = cv2.resize(frame1, (300, 200))

            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                # frame_for_face_ Frame where face is to BE Detected 
                frame_save_path = image_root_dir + "frame_for_face_"+ str(frame_counter)+"__.png"
                cv2.imwrite(frame_save_path, frame1) 
                frame_counter += 1
                last_save_time = current_time  # Update the last save time
                print("---TYPE--SAVED--frame_for_face_--",type(resized_frame1))
                face_write_path = FaceDetection().face_detect_yolo_huggin_face(frame_save_path)
                print("----face_write_path--->> %s",face_write_path)
                if isinstance(face_write_path,str): # TODO == ##TODO_9thMARCHMARCH--this face_write_path, will RETURN -as - NONE 
                    #
                    


                # print("---TYPE--AA--",type(resized_frame1))
                # print("---PATH---image_saved_last-",frame_save_path) # TODO -- get here Processed Frames
                # image_saved_last = cv2.imread(frame_save_path)
                # print("---PATH---image_saved_last-",type(image_saved_last))

            # ret2, frame2 = cap_tapo_2.read()
            # ret3, frame3 = cap_tapo_3.read()
            # ret4, frame4 = cap_tapo_4.read()
            # #

            # #Create a 2x2 grid of frames
            # if isinstance(image_saved_last,np.ndarray):
            #     print("--Image_Saved_Last---[ALERT]--->>")#,type(image_saved_last))
            #     print("   "*200)
            #     # Get image dimensions
            #     height, width, _ = image_saved_last.shape

            #     # Define rectangle coordinates (top-left and bottom-right corners)
            #     # Ensure the rectangle stays within the image boundaries
            #     x1, y1 = 0, 10  # Top-left corner
            #     x2, y2 = 300, 300  # Bottom-right corner

            #     # Clamp the coordinates to stay within the image boundaries
            #     x1 = max(0, min(x1, width - 1))
            #     y1 = max(0, min(y1, height - 1))
            #     x2 = max(0, min(x2, width - 1))
            #     y2 = max(0, min(y2, height - 1))

            #     # Draw the rectangle on the image
            #     color = (0, 255, 0)  # Green color in BGR format
            #     color_alert = (0, 0, 255)  # (255, 0, 0)  ## BLUE 
            #     thickness = 20 # Thickness of the rectangle border
            #     thickness_1 = 24 # Thickness of the rectangle border
            #     image_saved_last_copy = image_saved_last.copy()
            #     cv2.rectangle(image_saved_last, (x1, y1), (x2, y2), color, thickness)
            #     cv2.rectangle(image_saved_last_copy, (x1, y1), (x2, y2), color_alert, thickness_1)

            #     top_row = np.hstack((resized_frame1, image_saved_last)) #print("---TYPE--top_row--",type(top_row))
            #     cv2.imshow('OVERLANDER_TECH__VIEW_ALERT__', image_saved_last_copy)
            # else:
            #     top_row = np.hstack((resized_frame1, resized_frame1)) #print("---TYPE--top_row--",type(top_row))
            
            # bottom_row = np.hstack((resized_frame1, resized_frame1))
            #bottom_row_1 = np.hstack((resized_frame1, resized_frame1))

            # grid = np.vstack((top_row, bottom_row))
            # grid_1 = np.vstack((bottom_row,top_row))
            # #
            # # Display the resulting grid
            # cv2.imshow('OVERLANDER_TECH__GRID_VIEW', grid)
            # cv2.imshow('OVERLANDER_TECH__GRID_2', grid_1)
            

            # # Press 'q' to quit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # # Release the capture objects and close all windows
        # cap_tapo_1.release()
        # # cap_tapo_2.release()
        # # cap_tapo_3.release()
        # # cap_tapo_4.release()
        # cv2.destroyAllWindows()














""" 
#     list_of_files = os.listdir('log')
#     full_path = ["log/{0}".format(x) for x in list_of_files]

#     if len(list_of_files) == 15:
#         oldest_file = min(full_path, key=os.path.getctime)
#         os.remove(oldest_file)

# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break
"""