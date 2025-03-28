from datetime import datetime
import os
from util_logger import setup_logger_linux
logger = setup_logger_linux(module_name=str(__name__))

import mediapipe as mp
from mediapipe.tasks import python as media_pipe_python_api
from mediapipe.tasks.python import vision as media_pipe_vision_api


import cv2
import mediapipe as mp
import numpy as np
#@markdown To better demonstrate the Pose Landmarker API, we have created a set of visualization tools that will be used in this colab. These will draw the landmarks on a detect person, as well as the expected connections between those markers.
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#mp_pose = mp.solutions.pose

class MediaPipeGoog():
  """ 
  """

  @classmethod
  def get_horizontal_line_position(self,height, width):
    """ 
    """
    # Calculate the Y-coordinate for the horizontal line (30% of the height)
    line_y = int(height * 0.70) #TODO# 0.30 -- from TOP 30% --horizontal line (30% of the height)
    # Draw the horizontal line
    return line_y

  @classmethod
  def draw_horizontal_line(self,pose_annotated_image):
    """ 
    """
    # Get image dimensions
    height, width, _ = pose_annotated_image.shape
    
    # Calculate the Y-coordinate for the horizontal line (30% of the height)
    line_y = int(height * 0.70) #TODO# 0.30 -- from TOP 30% --horizontal line (30% of the height)
    # Draw the horizontal line
    line_color = (0, 255, 0)  # Green color in BGR format
    line_thickness = 5  # Thickness of the line
    cv2.line(pose_annotated_image, (0, line_y), (width - 1, line_y), line_color, line_thickness)
    # Print the pixel coordinates of the line
    print(f"Line coordinates:")
    print(f"Start: (0, {line_y})")
    print(f"End: ({width - 1}, {line_y})")
    """ 
    Line coordinates:
        Start: (0, 324)
        End: (1919, 324)

    """
    logger.debug("--pose_annotated_image--AAbb---line_y--> %s" ,line_y)
    return pose_annotated_image

  @classmethod
  def pose_draw_landmarks_on_image(self,
                                   rgb_image, 
                                   detection_result):
    """ 
    def draw_landmarks_on_image(rgb_image, detection_result):
    """
    flag_pose_landamrks_detected = "YES_LANDMARKS"
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    pose_landmarks_list = detection_result.pose_landmarks
    print("--LEN-LIST---pose_landmarks_list-",len(pose_landmarks_list))
    print("   "*200)
    if len(pose_landmarks_list) <1:
       flag_pose_landamrks_detected = "NO_LANDMARKS"
       return annotated_image , flag_pose_landamrks_detected

    line_y_coord = self.get_horizontal_line_position(height, width)
    text_color = (0, 255, 0)
    text_color_1 = (0,0,255)
    font = cv2.FONT_HERSHEY_PLAIN  # Thinnest font available
    font_scale = 10 # 0.5 == Smaller font scale for thinner appearance
    font_scale_alert = 2 # 0.5 == Smaller font scale for thinner appearance
    thickness = 3 # 1 -- Thinnest possible thickness
    
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx] # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
                ])
        ## TODO -- Ok Dont Draw Landmarks from MEDIAPIPE -- Only Draw OpenCV CIRCLES as below 
        # solutions.drawing_utils.draw_landmarks(
        #         annotated_image,
        #         pose_landmarks_proto,
        #         solutions.pose.POSE_CONNECTIONS,
        #         solutions.drawing_styles.get_default_pose_landmarks_style()
        #         )
        
        # Draw landmark numbers
        #ls_hand_only_landmarks = [27,29,31] 
        for landmark_idx, landmark in enumerate(pose_landmarks):
            if landmark_idx == 5 :#or landmark_idx == 6 :#or landmark_idx == 31: 
                ##TODO # if landmark_idx in ls_hand_only_landmarks:
                x = int(landmark.x * width) # Get the landmark position in pixel coordinates
                y_coord_height = int(landmark.y * height)
                
                if line_y_coord > y_coord_height:
                    line_color_2 = (0, 0, 255)  # RED - color in BGR format
                    line_thickness = 5  # Thickness of the line

                    # Define the point coordinate (center of the square)
                    point_x, point_y = x, y_coord_height  # Replace with your point coordinates
                    # Define the height and width of the square
                    square_height = 55
                    square_width = 35
                    # Calculate the top-left and bottom-right corners of the square
                    top_left_x = point_x - square_width // 2
                    top_left_y = point_y - square_height // 2
                    bottom_right_x = point_x + square_width // 2
                    bottom_right_y = point_y + square_height // 2
                    cv2.rectangle(annotated_image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), line_color_2, line_thickness)

                    #Draw the horizontal line
                    #cv2.line(annotated_image, (0, y_coord_height), (width - 1, 50), line_color_2, line_thickness)
                    cv2.putText(annotated_image, "-FACE-", (x, y_coord_height), font, font_scale_alert, text_color_1, thickness)

                    # #cv2.line(image, (0, y_coord_height), (width - 1, y_coord_height), line_color, line_thickness)

                    # logger.debug("-AAbb----LINE--CROSSED---landmark_idx--> %s" ,landmark_idx)
                    # logger.debug("-AAbb------LINE--CROSSED---landmark---> %s" ,landmark)
                    # logger.debug("-AAbb------LINE--CROSSED---landmark.y----> %s" ,int(landmark.y))

                # if line_y_coord < y_coord_height:
                #     crossed_landmark_idx = str(landmark_idx) + "--CROSS-AA"
                #     cv2.putText(annotated_image, str(crossed_landmark_idx), (x, y_coord_height), font, font_scale_alert, text_color_1, thickness)
                #     print("   -- "*10)
                #     print("--LINE--CROSSED----------------------------------------")
                #     print("   -- "*10)
                #     logger.debug("--LINE--CROSSED---landmark_idx--> %s" ,landmark_idx)
                #     logger.debug("--LINE--CROSSED---line_y_coord--> %s" ,line_y_coord)
                #     logger.debug("--LINE--CROSSED---landmark.y--> %s" ,int(landmark.y))
                #     logger.debug("--LINE--CROSSED---y_coord_height--> %s" ,y_coord_height)
                
                # else:
                #    continue

                # Draw the landmark number
                # logger.debug("--a--landmark_idx--> %s" ,landmark_idx)
                # cv2.putText(annotated_image, str(landmark_idx), (x, y_coord_height), font, font_scale, text_color, thickness)
                # # #cv2.putText(annotated_image, str(landmark_idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                # cv2.circle(annotated_image, (x, y_coord_height), 7, text_color_1, -1)  # Green circle with radius 5

        annotated_image = self.draw_horizontal_line(annotated_image) #
        # TODO -- # OK Dont Draw Horizonta Line -- Keep it INVISIBLE 
        return annotated_image , flag_pose_landamrks_detected

  @classmethod
  def pose_media_pipe_google_2(self):
    """
    Desc:
      - For Static Images -- Not IPWEBCAM Feed 
      - This is INIT Frame -- frame_pose_save_path == image_saved_ipcam
    """
    import time
    dir_pose_init_video = "../data_dir/pose_detected/init_video/"
    os.makedirs(dir_pose_init_video, exist_ok=True)  # Create 
    dir_pose_detected_pose = "../data_dir/pose_detected/detected_pose/"
    os.makedirs(dir_pose_detected_pose, exist_ok=True)  # Create the output directory if it doesn't exist

    # frame_counter = 0
    # save_interval = 1  # Save a frame every 2 seconds
    # last_save_time = time.time()
    
    dir_pose_init_video = "../data_dir/pose_detected/init_video/"
    #init_vid_name = "Columbia_.mp4"
    init_vid_name = "Bangladesh_.mp4"
    #init_vid_name = "DELHI_RAILWAY_.mp4" # NOT OK 
    #init_vid_name = "vegas_1.mp4" # OK 
    #init_vid_name = "METRO_.mp4" # OK 
    #init_vid_name = "germanpolice_.mp4"
    
    
    
    capture_vid_init = cv2.VideoCapture(dir_pose_init_video+init_vid_name)
    # Check if the video was opened successfully
    if not capture_vid_init.isOpened():
        print(f"Error: Unable to open video file--")
        exit()
    # Get video properties
    fps = capture_vid_init.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_count = int(capture_vid_init.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    duration_of_video = frame_count / fps  # Total duration of the video in SECONDS
    print(f"Video FPS: {fps}")
    print(f"Total Frames: {frame_count}")
    print(f"Video Duration: {duration_of_video:.2f} seconds")
    # Frame capture interval (every 2 seconds)
    capture_interval = 1  # in seconds
    frame_interval = int(fps * capture_interval)  # Number of frames to skip
    # Initialize variables
    current_frame = 0
    captured_frame_count = 0
    while True: # Loop through the video frames
        ret, frame = capture_vid_init.read()  # Read the next frame
        resized_frame_0 = cv2.resize(frame, (500,650)) 
        if not ret: # Break the loop if no more frames are available
            break
        if current_frame % frame_interval == 0: # Capture a frame every 2 seconds
            resized_frame1 = cv2.resize(frame, (500,650)) 
            # Save the captured frame
            frame_pose_save_path = os.path.join(dir_pose_detected_pose, f"frame_{captured_frame_count:04d}.jpg")
            cv2.imwrite(frame_pose_save_path, resized_frame1)
            print(f"Frame {captured_frame_count} saved to {frame_pose_save_path}")
            captured_frame_count += 1
            pose_write_path = MediaPipeGoog().pose_media_pipe_google_0(frame_pose_save_path)
            if isinstance(pose_write_path,str):
              if pose_write_path == "EMPTY_STR":
                continue # TODO -- get OLder -- pose_write_path --
              image_pose_saved_last = cv2.imread(pose_write_path)
              resized_pose_frame = cv2.resize(image_pose_saved_last, (500,650)) 
              top_row = np.hstack((resized_frame_0, resized_pose_frame))
              cv2.imshow('OVERLANDER__GRID_VIEW', top_row) # TODO -- grid
              # Press 'q' to quit
              if cv2.waitKey(1) & 0xFF == ord('q'):
                  break

        # Increment the frame counter
        current_frame += 1

    # Release the video capture object
    capture_vid_init.release()
    cv2.destroyAllWindows()
    print("Frame capture completed.")
    

    # #while True:
    # ret1, frame1 = capture_vid_init.read() 
    # logger.warning("-pose---Abb --frame1->> %s",type(frame1))
    # current_time = time.time()
    # if current_time - last_save_time >= save_interval:
    #   logger.warning("-pose-Abb-current_time---> %s",current_time)

    #   resized_frame1 = cv2.resize(frame1, (900,850)) #TODO # 300 , 200 
    #   logger.warning("-pose---Abb --resized_frame1->> %s",type(resized_frame1))
    #   frame_pose_save_path = dir_pose_detected_pose + "frame_for_pose_"+ str(frame_counter)+"__.png" # frame_for_pose_ Frame where POSE is to BE Detected   
    #   cv2.imwrite(frame_pose_save_path, resized_frame1) 
    #   frame_counter += 1
    #   last_save_time = current_time  # Update the last save time


    

    # pose_write_path = "EMPTY_STR"
    # frame_counter = 0
    # dt_time_now = datetime.now()
    # time_minute_now  = dt_time_now.strftime("_%m_%d_%Y_%H_%M_%S")  
    # print("-time_minute_now----->> %s",time_minute_now)
    # second_now = str(time_minute_now).rsplit("_",1)[1]
    # print("-time_minute_now--min_now--->> %s",second_now)
    
    # # logger.warning("-POSE---TYPE---image_saved_ipcam->> %s",image_saved_ipcam)
    # # if "frame_for_pose" in str(image_saved_ipcam):
    # #     image_name_pose_detect = str(str(image_saved_ipcam).rsplit("frame_for_pose/",1)[1])
    # # ##../data_dir/pose_detected/frame_for_pose/frame_for_pose_0__.png

    # base_options = media_pipe_python_api.BaseOptions(model_asset_path='../data_dir/pose_detected/pose_models/pose_landmarker.task')
    # options =  media_pipe_vision_api.PoseLandmarkerOptions(
    #                             base_options=base_options,
    #                             output_segmentation_masks=True)
    #                             #running_mode=VisionRunningMode.IMAGE) ## TODO 
    # detector = media_pipe_vision_api.PoseLandmarker.create_from_options(options) ##- <class 'mediapipe.python._framework_bindings.image.Image'>
    # #image = mp.Image.create_from_file(image_saved_ipcam) #print("----mp.Image.create_from_file(image_rootDIR)----",type(image)) #
    
    # detection_result = detector.detect(image)
    # annotated_image = self.pose_draw_landmarks_on_image(image.numpy_view(), detection_result)
    
    # # WRITE TO DIR -- pose_rect_only
    # # name_to_write = image_name_pose_detect+"_frame_pose_"+str(second_now)+"__"+str(frame_counter)+"__.png"
    # # pose_write_path = dir_pose_rect_only+name_to_write
    # cv2.imwrite(pose_write_path, annotated_image)
    # frame_counter += 1
    # logger.warning("--pose_write_path-POSE FRAMES WRITTEN--- %s",pose_write_path)
    # print("----pose_write_path--AA->> %s",pose_write_path)
    # return pose_write_path

  @classmethod
  def pose_media_pipe_google_0(self,image_saved_ipcam):
    """
    This is INIT Frame -- frame_pose_save_path == image_saved_ipcam
    """
    logger.warning("-HIT-pose_media_pipe_google_0--->>")

    dir_pose_not_ipcam = "../data_dir/pose_detected/pose_not_ipcam/"
    dir_got_pose_id_not_ipcam = "../data_dir/pose_detected/pose_id_not_ipcam/" #pose_id_not_ipcam

    pose_write_path = "EMPTY_STR"
    frame_counter = 0
    dt_time_now = datetime.now()
    time_minute_now  = dt_time_now.strftime("_%m_%d_%Y_%H_%M_%S")  
    print("-time_minute_now----->> %s",time_minute_now)
    second_now = str(time_minute_now).rsplit("_",1)[1]
    print("-time_minute_now--min_now--->> %s",second_now)
    
    logger.warning("-pose_media_pipe_google_0--image_saved_ipcam->> %s",image_saved_ipcam)
    if "detected_pose" in str(image_saved_ipcam):
        image_name_pose_detect = str(str(image_saved_ipcam).rsplit("detected_pose/",1)[1])
    
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    base_options = python.BaseOptions(model_asset_path='../data_dir/pose_detected/pose_models/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
                                base_options=base_options,
                                output_segmentation_masks=True)
                                #running_mode=VisionRunningMode.IMAGE) ## TODO 
    detector = vision.PoseLandmarker.create_from_options(options) ##- <class 'mediapipe.python._framework_bindings.image.Image'>
    image = mp.Image.create_from_file(image_saved_ipcam) #print("----mp.Image.create_from_file(image_rootDIR)----",type(image)) #
    detection_result = detector.detect(image)
    annotated_image , flag_pose_landamrks_detected = self.pose_draw_landmarks_on_image(image.numpy_view(), detection_result)
    ##pose_id_not_ipcam
    if flag_pose_landamrks_detected == "YES_LANDMARKS":
      dir_got_pose_id_not_ipcam
      name_to_write = image_name_pose_detect+"_frame_pose_"+str(second_now)+"__"+str(frame_counter)+"__.png"
      pose_write_path = dir_got_pose_id_not_ipcam+name_to_write
      cv2.imwrite(pose_write_path, annotated_image)
      frame_counter += 1
      logger.warning("---YES_LANDMARKS-pose_write_path-POSE FRAMES WRITTEN--- %s",pose_write_path)
      print("--YES_LANDMARKS--pose_write_path--AA->> %s",pose_write_path)
      return pose_write_path
    else:
      # WRITE TO DIR -- pose_rect_only
      name_to_write = image_name_pose_detect+"_frame_pose_"+str(second_now)+"__"+str(frame_counter)+"__.png"
      pose_write_path = dir_pose_not_ipcam+name_to_write
      cv2.imwrite(pose_write_path, annotated_image)
      frame_counter += 1
      logger.warning("--pose_write_path-POSE FRAMES WRITTEN--- %s",pose_write_path)
      print("----pose_write_path--AA->> %s",pose_write_path)
      return pose_write_path



  @classmethod
  def pose_media_pipe_google_1(self,image_saved_ipcam):
    """
    This is INIT Frame -- frame_pose_save_path == image_saved_ipcam
    """
    dir_pose_rect_only = "../data_dir/pose_detected/pose_rect_only/"
    pose_write_path = "EMPTY_STR"
    frame_counter = 0
    dt_time_now = datetime.now()
    time_minute_now  = dt_time_now.strftime("_%m_%d_%Y_%H_%M_%S")  
    print("-time_minute_now----->> %s",time_minute_now)
    second_now = str(time_minute_now).rsplit("_",1)[1]
    print("-time_minute_now--min_now--->> %s",second_now)
    
    logger.warning("-POSE---TYPE---image_saved_ipcam->> %s",image_saved_ipcam)
    if "frame_for_pose" in str(image_saved_ipcam):
        image_name_pose_detect = str(str(image_saved_ipcam).rsplit("frame_for_pose/",1)[1])
    ##../data_dir/pose_detected/frame_for_pose/frame_for_pose_0__.png
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    base_options = python.BaseOptions(model_asset_path='../data_dir/pose_detected/pose_models/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
                                base_options=base_options,
                                output_segmentation_masks=True)
                                #running_mode=VisionRunningMode.IMAGE) ## TODO 
    detector = vision.PoseLandmarker.create_from_options(options) ##- <class 'mediapipe.python._framework_bindings.image.Image'>
    image = mp.Image.create_from_file(image_saved_ipcam) #print("----mp.Image.create_from_file(image_rootDIR)----",type(image)) #
    
    detection_result = detector.detect(image)
    annotated_image = self.pose_draw_landmarks_on_image(image.numpy_view(), detection_result)
    # WRITE TO DIR -- pose_rect_only
    name_to_write = image_name_pose_detect+"_frame_pose_"+str(second_now)+"__"+str(frame_counter)+"__.png"
    pose_write_path = dir_pose_rect_only+name_to_write
    cv2.imwrite(pose_write_path, annotated_image)
    frame_counter += 1
    logger.warning("--pose_write_path-POSE FRAMES WRITTEN--- %s",pose_write_path)
    print("----pose_write_path--AA->> %s",pose_write_path)
    return pose_write_path



#   @classmethod
#   def media_pipe_google():
#     """ 
#     """
#     # get POSE Object 
#     obj_pose_static_img = mp_pose.Pose(
#                                     static_image_mode=True,
#                                     model_complexity=2,
#                                     enable_segmentation=True,
#                                     min_detection_confidence=0.5) 
#     print("--Type---",type(obj_pose_static_img))
#     print("--Type---",obj_pose_static_img)




# # For static images:
# IMAGE_FILES = []
# BG_COLOR = (192, 192, 192) # gray 
# with mp_pose.Pose(
#     static_image_mode=True,
#     model_complexity=2,
#     enable_segmentation=True,
#     min_detection_confidence=0.5) as pose:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     image_height, image_width, _ = image.shape
#     # Convert the BGR image to RGB before processing.
#     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     if not results.pose_landmarks:
#       continue
#     print(
#         f'Nose coordinates: ('
#         f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
#         f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
#     )

#     annotated_image = image.copy()
#     # Draw segmentation on the image.
#     # To improve segmentation around boundaries, consider applying a joint
#     # bilateral filter to "results.segmentation_mask" with "image".
#     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#     bg_image = np.zeros(image.shape, dtype=np.uint8)
#     bg_image[:] = BG_COLOR
#     annotated_image = np.where(condition, annotated_image, bg_image)
#     # Draw pose landmarks on the image.
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
#     # Plot pose world landmarks.
#     mp_drawing.plot_landmarks(
#         results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# # For webcam input:
# cap = cv2.VideoCapture(0)
# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as pose:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image)

#     # Draw the pose annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()