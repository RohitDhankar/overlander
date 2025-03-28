from ultralytics import FastSAM
import json
import time
import os , numpy as np , cv2
from util_logger import setup_logger_linux
logger = setup_logger_linux(module_name=str(__name__))


class FastSAMProcessor:
    """ 
    """

    @classmethod
    def import_fast_sam(self):
        """
        Import the FastSAM model using the specified weights.
        """
        try:
            model_weights="FastSAM-s.pt"
            model_fastSAM = FastSAM(model_weights)
            logger.warning("--TYPE--model_fastSAM---> %s",type(model_fastSAM))
            return model_fastSAM
            
        except Exception as e:
            print(f"Error loading FastSAM model: {e}")

    @classmethod
    def process_images_from_directory(self):
        """
        Read images from a directory, process them using FastSAM, and save the results.

        Args:## NONE -- , image_dir, output_dir="output", conf=0.4, iou=0.9
            image_dir (str): Path to the directory containing images.
            output_dir (str): Path to the directory to save processed images.
            conf (float): Confidence threshold for inference.
            iou (float): IoU threshold for inference.
        """
        res_image_save_path = "EMPTY_STR"
        
        model_fastSAM = self.import_fast_sam()
        logger.warning("--TYPE--model_fastSAM--AA-> %s",type(model_fastSAM))
        dir_fast_sam_init_video = "../data_dir/fast_sam/init_video/"
        os.makedirs(dir_fast_sam_init_video, exist_ok=True)  
        dir_detected_segs = "../data_dir/fast_sam/detected_segs/"
        os.makedirs(dir_detected_segs, exist_ok=True)
        dir_res_detected_segs = "../data_dir/fast_sam/res_detected_segs/"
        os.makedirs(dir_res_detected_segs, exist_ok=True)
        
        dir_pose_init_video = "../data_dir/pose_detected/init_video/"
        #init_vid_name = "Columbia_.mp4"
        #init_vid_name = "Bangladesh_.mp4"
        #init_vid_name = "DELHI_RAILWAY_.mp4" # NOT OK 
        init_vid_name = "vegas_1.mp4" # OK 
        #init_vid_name = "METRO_.mp4" # OK 
        #init_vid_name = "germanpolice_.mp4"

        # Initialize variables
        current_frame = 0
        captured_frame_count = 0

        save_interval = 1  # Save a frame every 3 seconds
        last_save_time = time.time()

        capture_vid_init = cv2.VideoCapture(dir_pose_init_video+init_vid_name)
        # Check if the video was opened successfully
        # if not capture_vid_init.isOpened():
        #     print(f"Error: Unable to open video file--")
        #     exit()
        # Get video properties
        # fps = capture_vid_init.get(cv2.CAP_PROP_FPS)  # Frames per second
        # frame_count = int(capture_vid_init.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
        # duration_of_video = frame_count / fps  # Total duration of the video in SECONDS
        # print(f"Video FPS: {fps}")
        # print(f"Total Frames: {frame_count}")
        # print(f"Video Duration: {duration_of_video:.2f} seconds")
        # Frame capture interval (every 2 seconds)
        # capture_interval = 0.25  # in seconds
        # frame_interval = int(fps * capture_interval)  # Number of frames to skip


        while True: # Loop through the video frames
            ret, frame = capture_vid_init.read()  # Read the next frame
            # if not ret: # Break the loop if no more frames are available
            #     break
            resized_frame_0 = cv2.resize(frame, (500,650)) 
            current_time = time.time()

            if current_time - last_save_time >= save_interval: ## earlier -- #if current_frame % frame_interval == 0: 
                #
                resized_frame1 = cv2.resize(frame, (500,650)) 
                # Save the captured frame
                print(f"---captured_frame_count--GHGH-->> {captured_frame_count}")
                logger.warning("---model_fastSAM--GHGHG--captured_frame_count----> %s",captured_frame_count)
                #frame_pose_save_path = os.path.join(dir_detected_segs,"_frame_{captured_frame_count:04d}.png")
                frame_pose_save_path = dir_detected_segs+init_vid_name+str(captured_frame_count)+"_.png"
                cv2.imwrite(frame_pose_save_path, resized_frame1)
                #print(f"Frame {captured_frame_count} saved to {frame_pose_save_path}")

                last_save_time = current_time  # Update the last save time

                captured_frame_count += 1
                device_SAM = 0 # GPU - device=0

                # results = model_fastSAM(frame_pose_save_path, device=device_SAM, conf=0.4, iou=0.9)
                # logger.warning("---model_fastSAM--AABB--TYPE(results----> %s",type(results)) ## LIST 
                # logger.warning("---model_fastSAM--AABB--TYPE(results----> %s",type(results[0]))
                # logger.warning("---model_fastSAM--results----> %s",results)
                # logger.warning("---model_fastSAM--results-0---> %s",results[0])
                # str_res_json = results[0].to_json()
                # logger.warning("---model_fastSAM--AAbbcc--str_res_json---> %s",str_res_json)
                
                # # # Run inference with texts prompt
                results_person = model_fastSAM(frame_pose_save_path, texts="image of person",device=device_SAM, conf=0.4, iou=0.9)
                #logger.warning("---model_fastSAM--GHGH-results_person-0--> %s",type(results_person[0]))
                str_res_json = results_person[0].to_json()
                #logger.warning("---model_fastSAM--AA-MMM--str_res_json---> %s",str_res_json)
                #python convert JSON to DICT 
                dict_res_json = json.loads(str_res_json)
                #logger.warning("---model_fastSAM--BNNN-str_res_json---> %s",type(dict_res_json[0]))
                if float(dict_res_json[0]["confidence"]) >= 0.8:
                    #logger.warning("---model_fastSAM--AAbbddff--GOT-80---dict_res_json[confidence]---> %s",dict_res_json[0]["confidence"])
                    res_segs_img_name = "FastSAM_person_80_" +str(captured_frame_count) + "_.png"
                    res_image_save_path = os.path.join(dir_res_detected_segs,res_segs_img_name)
                    results_person[0].save(res_image_save_path)
                if isinstance(res_image_save_path,str):
                    if res_image_save_path == "EMPTY_STR":
                        continue # TODO -- get OLder -- res_image_save_path --
                    ## for Grid 
                    logger.warning("--EMPTY-STR--res_image_save_path---> %s",res_image_save_path)
                    image_pose_saved_last = cv2.imread(res_image_save_path)
                    resized_pose_frame = cv2.resize(image_pose_saved_last, (500,650)) 
                    top_row = np.hstack((resized_frame_0, resized_pose_frame))
                    cv2.imshow('OVERLANDER__GRID_VIEW', top_row) # TODO -- grid
                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        capture_vid_init.release()
        cv2.destroyAllWindows()
        print("Frame capture completed.")

                



        # # Process each image
        # for image_file in image_files:
        #     image_path = os.path.join(image_dir, image_file)
        #     print(f"Processing image: {image_path}")

        #     # Run inference on the image
        #     results = self.model(image_path, device=self.device, conf=conf, iou=iou)

        #     # Save the results
        #     output_path = os.path.join(output_dir, f"processed_{image_file}")
        #     results[0].save(output_path)
        #     print(f"Processed image saved to: {output_path}")

# # Example usage
# # Define an inference source
# source = "path/to/bus.jpg"

# # Create a FastSAM model
# model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

# # Run inference on an image
# everything_results = model(source, device="cpu", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

# # Run inference with bboxes prompt
# results = model(source, bboxes=[439, 437, 524, 709])

# # Run inference with points prompt
# results = model(source, points=[[200, 200]], labels=[1])

# # Run inference with texts prompt
# results = model(source, texts="a photo of a dog")

# # Run inference with bboxes and points and texts prompt at the same time
# results = model(source, bboxes=[439, 437, 524, 709], points=[[200, 200]], labels=[1], texts="a photo of a dog")

# def __init__(self, model_weights="FastSAM-s.pt", device="cpu"):
#     """
#     Initialize the FastSAMProcessor class.

#     Args:
#         model_weights (str): Path to the FastSAM model weights file.
#         device (str): Device to run the model on (e.g., "cpu" or "cuda").
#     """
#     self.model_weights = model_weights
#     self.device = device
#     self.model = None

# # Get list of image files in the directory
# image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
# if not image_files:
#     print(f"No images found in directory: {image_dir}")
#     return    