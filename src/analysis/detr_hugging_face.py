
from datetime import date
from datetime import datetime

from transformers import (DetrImageProcessor, 
                            DetrForObjectDetection,
                            AutoImageProcessor,
                            AutoModelForObjectDetection)

from transformers import pipeline
checkpoint = "PekingU/rtdetr_v2_r50vd"
pipeline_rtdetr_v2 = pipeline("object-detection", model=checkpoint, image_processor=checkpoint)

#DetrFeatureExtractor) ##DetrFeatureExtractor ## Deprecated warning 
                            
from util_logger import setup_logger_linux
logger = setup_logger_linux(module_name=str(__name__))
import torch , os , cv2 
import numpy as np
import supervision as sv
from PIL import Image
import requests ,math
import matplotlib.pyplot as plt ##Error--plot_results-

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

#%config InlineBackend.figure_format = 'retina'

# import ipywidgets as widgets
# from IPython.display import display, clear_output

# import torch
# from torch import nn
# from torchvision.models import resnet50
# import torchvision.transforms as T
# torch.set_grad_enabled(False);

# # you can specify the revision tag if you don't want the timm dependency
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# you can specify the revision tag if you don't want the timm dependency
detr_image_processor_detr_resnet101 = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
logger.warning("--init---detr_image_processor_detr_resnet101--> %s" ,type(detr_image_processor_detr_resnet101))
model_detr_resnet101 = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
logger.warning("--init---model_detr_resnet101---> %s" ,type(model_detr_resnet101))
#
detr_image_processor_detr_resnet101_dc5 = DetrImageProcessor.from_pretrained('facebook/detr-resnet-101-dc5')
logger.warning("--init---detr_image_processor_detr_resnet101_dc5---> %s" ,type(detr_image_processor_detr_resnet101_dc5))
model_detr_resnet101_dc5 = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-dc5')
logger.warning("--init---model_detr_resnet101_dc5---> %s" ,type(model_detr_resnet101_dc5))

# COCO classes
CLASSES = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

#from read_cam.read_webcam import CV2VideoCapture ## TODO - Dont CYCLIC IMPORT 

class GetFramesFromVids:
    """ 
    """

    @classmethod
    def get_static_vids_local_list(self):
        """ 
        """
        try:
            #init_vid_dir = CV2VideoCapture().get_init_dir() ## TODO - Dont CYCLIC IMPORT 
            logger.debug("--get_static_vids_local_list---init_vid_dir-----> %s" ,init_vid_dir)
            
            ## TODO -- testing with Old Videos DIR 
            #init_vid_dir = "../data_dir/init_vid_dir_2025-02-02-21_08_/" ##init_vid_dir_2025-02-02-21_08_
            #init_vid_dir = "../data_dir/init_vid_small_1/" # OK 
            init_vid_dir = "../data_dir/init_vid_dir_2025-02-02-21_08_/"
            ls_video_files_uploads = []
            for root, dirs, files in os.walk(init_vid_dir):
                for filename in files:
                    ls_video_files_uploads.append(os.path.join(root, filename))
            logger.debug("-List of VIDEO-Files----> %s" ,ls_video_files_uploads)
            return ls_video_files_uploads
        except Exception as err:
            logger.error("-Error--get_static_vids_local_list---> %s" ,err)

    @classmethod
    def get_frame_from_video(self):
        """
        """
        print(f"-get_frame_from_video---HIT--> ")
        try:
            logger.debug("-get_frame_from_video---HIT->" )

            root_dir = "../data_dir/out_vid_frames_dir/"
            ls_frames_to_write = [4,11,17,25,30,37,45,55,66,77,88,100,110]
            ##ls_video_files_uploads = self.get_static_vids_local_list() ## TODO - Dont CYCLIC IMPORT 
            for iter_vid in range(len(ls_video_files_uploads)):
                count = 0
                vid_short_name = str(ls_video_files_uploads[iter_vid])
                if "T" in str(vid_short_name):
                    vid_short_name = vid_short_name.rsplit("T",1)
                    print("---vid_short_name--a-TTT--\n",vid_short_name)
                    vid_short_name = str(str(vid_short_name[1]).rsplit("_",0))
                else:
                    pass
                if "mp4" in str(vid_short_name):
                    vid_short_name = vid_short_name.replace(".mp4","")
                else: # TODO - Video Format == #.webm
                    vid_short_name = vid_short_name.replace(".webm","")
                vid_short_name = vid_short_name.replace("['","")
                vid_short_name = vid_short_name.replace("']","")
                logger.debug("-Written--Video----> %s",vid_short_name)

                vidcap = cv2.VideoCapture(ls_video_files_uploads[iter_vid]) #eo_file__2025-01-26T11-32-45-853845_.mp4'
                count_of_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
                logger.debug("-TOTAL--COUNT--Of--FRAMES---> %s",count_of_frames)

                for iter_frame in range(len(ls_frames_to_write)):
                    if ls_frames_to_write[iter_frame] >= count_of_frames -2:
                        pass 
                    else:

                        vidcap.set(cv2.CAP_PROP_POS_FRAMES, ls_frames_to_write[iter_frame])
                        success, image = vidcap.read() #while success:
                        logger.debug("--WRITING-FRAME----type(image)--> %s",type(image)) ## Numpy nd.array 
                        # print('width of Image: ', image.shape[1]) ## Pixels -- width of Image:  1920
                        # print('height of Image:', image.shape[0]) ## Pixels 
                        print('Size of Image:', image.size) ## Pixels --- Size of Image: 6220800
                        if image.size >= 6000000: ##62,20,800 - MAX for JPEG -- 65,500
                            print('LArge-forJPEG-Size of Image:', image.size) ## Pixels --- Size of Image: 6220800
                            ## TODO -- Testing for JPEG Size Issues -- "__.jpg"

                            frame_save_path = root_dir + str(vid_short_name) + "_frame_"+ str(count)+"__.jpg" #"__.tif"
                        else:
                            frame_save_path = root_dir + str(vid_short_name) + "_frame_"+ str(count)+"__.jpg"

                        logger.debug("--WRITING-FRAME----> %s",frame_save_path)
                        #vidcap.release()
                        cv2.imwrite(frame_save_path, image) # save frame as .tif file      
                        count += 1
        except Exception as err:
            logger.error("-Error--get_frame_from_video---> %s" ,err)

class AnalysisVideoFrames:
    """ 
    """

    @classmethod
    def box_cxcywh_to_xyxy(self,x):     # for output bounding box post-processing
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    @classmethod
    def rescale_bboxes(self,out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

class PlotBboxOnFrames:
    """ 
    Desc:
        - Plot the Bounding Boxes on the Objects within each Image Frame
    """

    @classmethod
    def get_frames_local_list(self):
        """ 
        """
        try:
            root_dir = "../data_dir/out_vid_frames_dir/" #root_dir = "static/image_uploads/"
            ls_files_uploads = []
            for root, dirs, files in os.walk(root_dir):
                for filename in files:
                    ls_files_uploads.append(os.path.join(root, filename))
            logger.debug("-List of Image Files Uploaded ----> %s",ls_files_uploads)
            return ls_files_uploads
        except Exception as err:
            logger.error("-Error--get_frames_local_list---> %s" ,err)

    @classmethod
    def get_bbox_on_frames(self):
        """ 
        """
        try:
            ls_files_uploads = self.get_frames_local_list()
            for iter_img in range(len(ls_files_uploads)):
                image_name_for_zoom = ls_files_uploads[iter_img]
                image_name = str(ls_files_uploads[iter_img]).replace("../data_dir/out_vid_frames_dir/","")
                bbox_image_name = "bbox_"+ image_name
                logger.debug("--BBOX-IMAGE-NAME----> %s",bbox_image_name)
                image = Image.open(ls_files_uploads[iter_img])
                try:
                    inputs = detr_image_processor_detr_resnet101_dc5(images=image, return_tensors="pt") ## RESNET--101
                    outputs = model_detr_resnet101_dc5(**inputs)
                except Exception as err:
                    logger.error("-Error--detr_image_processor_detr_resnet101_dc5---> %s" ,err)

                probas = outputs.logits.softmax(-1)[0, :, :-1] ## ORIGINAL CODE 
                keep = probas.max(-1).values > 0.7 ## TODO -- keep any prediction with VALS 40% 
                # convert boxes from [0; 1] to image scales
                bboxes_scaled = AnalysisVideoFrames().rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
                self.plot_results(image, bbox_image_name,
                                    image_name_for_zoom,
                                    probas[keep], 
                                    bboxes_scaled)

                # convert outputs (bounding boxes and class logits) to COCO API
                # let's only keep detections with score > 0.9
                target_sizes = torch.tensor([image.size[::-1]])
                logger.debug("--target_sizes---> %s",target_sizes)
                results = detr_image_processor_detr_resnet101_dc5.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
                logger.debug("--ALL--Results---> %s",results)

                # logger.debug("--Lables within Results---> %s",results["labels"]) #print(results["labels"])
                # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                #     logger.debug("--BBOX--List within Results---> %s",box.tolist())
                #     logger.debug("--BBOX--[Label-Item] within Results---> %s",model_detr_resnet101_dc5.config.id2label[label.item()])
                #     BBox = [round(i, 2) for i in box.tolist()]
                #     logger.debug("--BBOX--[BBOX] within Results---> %s",BBox)
                #     # print(
                    #         f"Detected {model.config.id2label[label.item()]} with confidence "
                    #         f"{round(score.item(), 3)} at location {BBox}"
                    # )
                    #logger.debug("-Detected-BBOX-LABEL---> %s -List within Results---> %s",model.config.id2label[label.item()],box.tolist())
        except Exception as err:
            logger.error("-Error--get_bbox_on_frames---> %s" ,err)

    @classmethod
    def plot_results(self,
                     pil_img, 
                    bbox_image_name,
                    image_name_for_zoom,
                    prob, boxes):
        """ 
        Desc:
            - Method to Plot the Bounding Boxes on the Objects within each Image Frame
        """
        try:
            ls_car_images = []
            plt.figure(figsize=(25,25))
            plt.imshow(pil_img)
            ax = plt.gca()
            colors = COLORS * 100
            for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=c, linewidth=3)) ## Drawing_BBox
                cl = p.argmax()
                text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
                if "car" in str(text) or "person" in str(text) or "bottle" in str(text):
                    logger.debug("--BBOX-[CAR]-[PERSON]-[BOTTLE]--> %s",str(text))
                    ls_car_images.append(bbox_image_name)
                    # bbox_image_path = bbox_image_name 
                    #img_zoomed_and_cropped = self.zoom_center(image_name_for_zoom,zoom_factor=1.5)
                    root_dir = "../data_dir/zoomed_car_imgs/"
                    #cv2.imwrite(root_dir+bbox_image_name+'_zoomed.png', img_zoomed_and_cropped) # TODO -- save 
                    #cv2.imwrite(root_dir+bbox_image_name+'_zoomed.png', pil_img) # TODO -- save Original Not Zoomed 
                    ax.text(xmin, ymin, text, fontsize=15,bbox=dict(facecolor='yellow', alpha=0.5))
                    plt.axis('off')
                    root_save_dir = "../data_dir/bbox_image_saved/"
                    plt.savefig(root_save_dir+bbox_image_name, bbox_inches='tight') ## TODO -- save 
                    
                else:
                    pass ## Done save images without these BBOX -- 
            plt.close()
            return ls_car_images  
        except Exception as err:
            logger.error("-Error--plot_results---> %s" ,err)

    @classmethod
    def zoom_center(self,
                    img, 
                    zoom_factor=1.5):
        """ 
        get a zoomed image -- But centered
        """
        try:
            img = cv2.imread(img)
            y_size = img.shape[0]
            x_size = img.shape[1]
            x1 = int(0.5*x_size*(1-1/zoom_factor)) # define new boundaries
            x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))
            y1 = int(0.5*y_size*(1-1/zoom_factor))
            y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))
            img_cropped = img[y1:y2,x1:x2] # first crop image then scale
            img_zoomed_and_cropped = cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)
            return img_zoomed_and_cropped
        except Exception as err:
            logger.error("-Error--zoom_center---> %s" ,err)


# Define a mapping from RGB values to human-readable color names
rgb_color_mapping_1 = {
    (128, 128, 128): "grey",
    (0, 0, 255): "blue",
    (165, 42, 42): "brown",
    (255, 0, 0): "red",
    (0, 0, 0): "black",
    (255, 255, 255): "white"
}

# Predefined RGB color mapping (human-readable names)
rgb_color_mapping_a = {
    (128, 128, 128): "grey",
    (0, 0, 255): "blue",
    (165, 42, 42): "brown",
    (255, 0, 0): "red",
    (0, 0, 0): "black",
    (255, 255, 255): "white",
    (0, 255, 0): "lime",
    (0, 128, 0): "green",
    (255, 255, 0): "yellow",
    (255, 165, 0): "orange",
    (128, 0, 128): "purple",
    (255, 192, 203): "pink",
    (75, 0, 130): "indigo",
    (240, 230, 140): "khaki",
    (0, 255, 255): "cyan",
    (0, 128, 128): "teal",
    (139, 69, 19): "saddlebrown",
    (255, 140, 0): "darkorange",
    (220, 20, 60): "crimson",
    (255, 99, 71): "tomato",
    (255, 215, 0): "gold",
    (218, 165, 32): "goldenrod",
    (173, 216, 230): "lightblue",
    (135, 206, 250): "lightskyblue",
    (70, 130, 180): "steelblue",
    (0, 191, 255): "deepskyblue",
    (25, 25, 112): "midnightblue",
    (106, 90, 205): "slateblue",
    (123, 104, 238): "mediumslateblue",
    (147, 112, 219): "mediumpurple",
    (139, 0, 139): "darkmagenta",
    (255, 0, 255): "magenta",
    (199, 21, 133): "mediumvioletred",
    (219, 112, 147): "palevioletred",
    (255, 20, 147): "deeppink",
    (255, 105, 180): "hotpink",
    (255, 228, 196): "bisque",
    (255, 218, 185): "peachpuff",
    (210, 180, 140): "tan",
    (244, 164, 96): "sandybrown",
    (205, 133, 63): "peru",
    (210, 105, 30): "chocolate",
    (160, 82, 45): "sienna",
    (128, 0, 0): "maroon",
    (85, 107, 47): "darkolivegreen",
    (107, 142, 35): "olivedrab",
    (154, 205, 50): "yellowgreen",
    (50, 205, 50): "limegreen",
    (34, 139, 34): "forestgreen",
    (0, 100, 0): "darkgreen",
    (46, 139, 87): "seagreen",
    (102, 205, 170): "mediumaquamarine",
    (32, 178, 170): "lightseagreen",
    (0, 206, 209): "darkturquoise",
    (72, 209, 204): "mediumturquoise",
    (175, 238, 238): "paleturquoise",
    (95, 158, 160): "cadetblue",
    (176, 196, 222): "lightsteelblue",
    (119, 136, 153): "lightslategray",
    (112, 128, 144): "slategray",
    (47, 79, 79): "darkslategray",
    (245, 245, 220): "beige",
    (253, 245, 230): "oldlace",
    (255, 250, 240): "floralwhite",
    (255, 250, 250): "snow",
    (240, 255, 240): "honeydew",
    (245, 255, 250): "mintcream",
    (240, 248, 255): "aliceblue",
    (248, 248, 255): "ghostwhite",
    (240, 255, 255): "azure",
    (230, 230, 250): "lavender",
    (255, 240, 245): "lavenderblush",
    (255, 228, 225): "mistyrose",
    (250, 235, 215): "antiquewhite",
    (250, 240, 230): "linen",
    (255, 239, 213): "papayawhip",
    (255, 235, 205): "blanchedalmond",
    (255, 222, 173): "navajowhite",
    (255, 228, 181): "moccasin",
    (255, 248, 220): "cornsilk",
    (255, 250, 205): "lemonchiffon",
    (255, 245, 238): "seashell",
    (245, 245, 245): "whitesmoke",
    (220, 220, 220): "gainsboro",
    (211, 211, 211): "lightgray",
    (192, 192, 192): "silver",
    (169, 169, 169): "darkgray",
    (128, 128, 128): "gray",
    (105, 105, 105): "dimgray",
    (0, 0, 0): "black",
    (255, 255, 255): "white"
}


# Predefined RGB color mapping (human-readable names)
rgb_color_mapping = {
    (128, 128, 128): "grey",
    (0, 0, 255): "blue",
    (165, 42, 42): "brown",
    (255, 0, 0): "red",
    (0, 0, 0): "black",
    (255, 255, 255): "white",
    (0, 255, 0): "lime",
    (0, 128, 0): "green",
    (255, 255, 0): "yellow",
    (255, 165, 0): "orange",
    (128, 0, 128): "purple",
    (255, 192, 203): "pink",
    (75, 0, 130): "indigo",
    (240, 230, 140): "khaki",
    (0, 255, 255): "cyan",
    (0, 128, 128): "teal",
    (139, 69, 19): "saddlebrown",
    (255, 140, 0): "darkorange",
    (220, 20, 60): "crimson",
    (255, 99, 71): "tomato",
    (255, 215, 0): "gold",
    (218, 165, 32): "goldenrod",
    (173, 216, 230): "lightblue",
    (135, 206, 250): "lightskyblue",
    (70, 130, 180): "steelblue",
    (0, 191, 255): "deepskyblue",
    (25, 25, 112): "midnightblue",
    (106, 90, 205): "slateblue",
    (123, 104, 238): "mediumslateblue",
    (147, 112, 219): "mediumpurple",
    (139, 0, 139): "darkmagenta",
    (255, 0, 255): "magenta",
    (199, 21, 133): "mediumvioletred",
    (219, 112, 147): "palevioletred",
    (255, 20, 147): "deeppink",
    (255, 105, 180): "hotpink",
    (255, 228, 196): "bisque",
    (255, 218, 185): "peachpuff",
    (210, 180, 140): "tan",
    (244, 164, 96): "sandybrown",
    (205, 133, 63): "peru",
    (210, 105, 30): "chocolate",
    (139, 69, 19): "saddlebrown",
    (160, 82, 45): "sienna",
    (128, 0, 0): "maroon",
    (85, 107, 47): "darkolivegreen",
    (107, 142, 35): "olivedrab",
    (154, 205, 50): "yellowgreen",
    (50, 205, 50): "limegreen",
    (34, 139, 34): "forestgreen",
    (0, 100, 0): "darkgreen",
    (46, 139, 87): "seagreen",
    (102, 205, 170): "mediumaquamarine",
    (32, 178, 170): "lightseagreen",
    (0, 206, 209): "darkturquoise",
    (72, 209, 204): "mediumturquoise",
    (175, 238, 238): "paleturquoise",
    (95, 158, 160): "cadetblue",
    (176, 196, 222): "lightsteelblue",
    (119, 136, 153): "lightslategray",
    (112, 128, 144): "slategray",
    (47, 79, 79): "darkslategray",
    (245, 245, 220): "beige",
    (253, 245, 230): "oldlace",
    (255, 250, 240): "floralwhite",
    (255, 250, 250): "snow",
    (240, 255, 240): "honeydew",
    (245, 255, 250): "mintcream",
    (240, 248, 255): "aliceblue",
    (248, 248, 255): "ghostwhite",
    (240, 255, 255): "azure",
    (230, 230, 250): "lavender",
    (255, 240, 245): "lavenderblush",
    (255, 228, 225): "mistyrose",
    (250, 235, 215): "antiquewhite",
    (250, 240, 230): "linen",
    (255, 239, 213): "papayawhip",
    (255, 235, 205): "blanchedalmond",
    (255, 222, 173): "navajowhite",
    (255, 228, 181): "moccasin",
    (255, 248, 220): "cornsilk",
    (255, 250, 205): "lemonchiffon",
    (255, 245, 238): "seashell",
    (245, 245, 245): "whitesmoke",
    (220, 220, 220): "gainsboro",
    (211, 211, 211): "lightgray",
    (192, 192, 192): "silver",
    (169, 169, 169): "darkgray",
    (128, 128, 128): "gray",
    (105, 105, 105): "dimgray",
    (0, 0, 0): "black",
    (255, 255, 255): "white"
}

class FaceDetection:
    """ 
    """

    @classmethod
    def invoke_model_yolov8_face_detection(self):
        """ 
        """
        # download model
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        logger.debug("----model_path--> %s" ,model_path)
        model_yolov8_face_detection = YOLO(model_path) # load model
        logger.debug("---model_yolov8_face_detection--> %s" ,type(model_yolov8_face_detection))
        return model_yolov8_face_detection
    
    # @classmethod
    # def invoke_model_yolove_general(self):
    #     """ 
    #     TODO --  YOLOE -- Model ID's on -- hf_hub_download
    #         "yoloe-v8s",
    #         "yoloe-v8m",
    #         "yoloe-v8l",
    #         "yoloe-11s",
    #         "yoloe-11m",
    #         "yoloe-11l",
    #     ],
    #     value="yoloe-v8l",
    #     """
    #     # download model
    #     ## TODO --  from yoloe_ultralytics import YOLOE

    #     model_id="yoloe-v8l"
    #     filename = f"{model_id}-seg.pt" 
    #     #filename = f"{model_id}-seg-pf.pt"

    #     model_path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
    #     logger.debug("--model_yolove_general--model_path--> %s" ,model_path)
    #     model_yolove_general = YOLOE(model_path) # load model
    #     model_yolove_general.eval()
    #     model_yolove_general.to("cuda" if torch.cuda.is_available() else "cpu")

    #     logger.debug("---model_yolove_general--> %s" ,type(model_yolove_general))
    #     return model_yolove_general

    
    @classmethod
    def invoke_model_yolo_hugginface(self):
        """ 
        TODO --  
    
        """
        # download model
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        logger.debug("----model_path--> %s" ,model_path)
        model_yolov8_face_detection = YOLO(model_path) # load model
        logger.debug("---model_yolov8_face_detection--> %s" ,type(model_yolov8_face_detection))
        return model_yolov8_face_detection
    
    
    @classmethod
    def face_detect_yolo_huggin_face(self,image_frame_path):
        """ 
        # TODO - SOURCE -- https://supervision.roboflow.com/latest/detection/annotators/#supervision.annotators.core.RoundBoxAnnotator-functions
        """
        try:

            from PIL import Image, ImageDraw
            face_write_path = None
            frame_counter = 0
            dt_time_now = datetime.now()
            time_minute_now  = dt_time_now.strftime("_%m_%d_%Y_%H_%M_%S")  
            print("-time_minute_now----->> %s",time_minute_now)
            second_now = str(time_minute_now).rsplit("_",1)[1]
            print("-time_minute_now--min_now--->> %s",second_now)

            root_dir = "../data_dir/face_detected/"
            dir_face_rect_only= "../data_dir/face_detected/face_rect_only/"

            model_yolov8_face_detection = self.invoke_model_yolov8_face_detection() ## TODO -- Toggle for FACES
            #model_yolove_general = self.invoke_model_yolove_general() # TODO -- Toggle for FACES
            logger.debug("--image_frame_path--AAA-NNN---> %s" ,image_frame_path) ##../data_dir/jungle_images/input_DIR/guj_1.png
            
            if "frame_for_face" in str(image_frame_path):
                image_name_face_detect = str(str(image_frame_path).rsplit("frame_for_face/",1)[1])
            else:
                image_name_face_detect = str(str(image_frame_path).rsplit("input_DIR/",1)[1])
            
            logger.warning("--image_frame_pathBBB---model_yolov8_face_detection-- %s",image_name_face_detect)
            image_for_process = cv2.imread(image_frame_path) ## image_for_process = cv2.imdecode(Image.open(image_frame_path))
            #image_for_roi = Image.open(image_frame_path)

            output = model_yolov8_face_detection(Image.open(image_frame_path)) ## TODO -- Toggle for FACES
            #output = model_yolove_general(Image.open(image_frame_path)) # TODO Not face generic ---# TODO -- Toggle for FACES
            results_face_detect = Detections.from_ultralytics(output[0]) #detections = sv.Detections(...)

            logger.debug("--results_face_detect---> %s" ,results_face_detect)
            logger.debug("--results_face_detect--Detections--aa-> %s" ,results_face_detect.xyxy)
            logger.debug("--=AA==results_face_detect--Detections--aa---TYPE--> %s" ,type(results_face_detect.xyxy)) #<class 'numpy.ndarray'>
            ls_faces_coord = results_face_detect.xyxy.tolist()
            if len(ls_faces_coord)>=1: # TODO -- Ok one or More Faces Deceted 
                #obj_pil_imagedraw = ImageDraw.Draw(image_for_roi)
                for face_bbox_coords_detection in results_face_detect.xyxy:
                    x1, y1, x2, y2 = face_bbox_coords_detection # Extract bounding box coordinates
                    #cv2.rectangle(image_for_process, (int(y1), int(y2)), (int(x1), int(x2)), border_color, border_thickness)
                    #obj_pil_imagedraw.rectangle([x1, y1, x2, y2], outline="red", width=2) # Draw bounding box rectangle
                    rectangle_portion_only = image_for_process[int(y1):int(y2), int(x1):int(x2)] # Save the extracted rectangular portion as a separate image

                    logger.warning("--image_frame_pathBBB----rectangle_portion_only--AA-- %s",type(rectangle_portion_only))
                    name_to_write = image_name_face_detect+"_frame_face_"+str(second_now)+"__"+str(frame_counter)+"__.png"
                    face_write_path = dir_face_rect_only+name_to_write
                    cv2.imwrite(face_write_path, rectangle_portion_only)
                    frame_counter += 1
                    logger.warning("--image_frame_pathBBB--WRITTEN--- %s",image_name_face_detect)
                    print("----face_write_path--AA->> %s",face_write_path)
                    return face_write_path 
                    ## TODO == ##TODO_9thMARCHMARCH--this face_write_path, will RETURN -as - NONE 
                    ## TODO == ##TODO_9thMARCHMARCH--- Ok below -- Dont Run Too many Logs -- 
                    #self.process_indl_faces(face_bbox_coords_detection,rectangle_portion_only)
            else:
                ## if len(ls_faces_coord)==0: # TODO --NO Faces Deceted 
                return face_write_path
        except Exception as err:
            logger.error("--DETR-face_detect_yolo_huggin_face-> %s" ,err)


        # Show image with bounding boxes
        #image_for_roi.show() ## OK CODE -- Dont pop ups -##TODO_9thMARCHMARCH

        # TODO -- ##TODO_9thMARCHMARCH - OK Code -- Needs Fixing Meta Data Table 

        # image_face_detected = cv2.imread(image_frame_path)        
        # box_annotator = sv.BoxAnnotator(color=sv.Color.GREEN,thickness=3) ## TODO - color=blue -
        # annotated_frame = box_annotator.annotate(
        #                         scene=image_face_detected.copy(),
        #                         detections=results_face_detect)
        # logger.debug("--results_face_detect-----annotated_frame---> %s" ,type(annotated_frame))
        
        # TODO -- ##TODO_9thMARCHMARCH - OK Code -- Needs Fixing Meta Data Table 
        #image_with_meta_table = self.table_on_image_frame(annotated_frame) 
        # logger.debug("--results_face_detect-----image_frame_path---> %s" ,image_frame_path)
        # ##./data_dir/out_vid_frames_dir/21-08-21-070423__frame_6__.jpg
        # print("--type-----image_frame_path---",type(image_frame_path))
        # face_annotated_image_name = str(image_frame_path).rsplit("_",0)
        # print("--split---",face_annotated_image_name) # TODO -- 
        # logger.debug("--results_face_detect-----face_annotated_image_name---> %s" ,face_annotated_image_name)
        # image_face_detect = str(str(face_annotated_image_name).rsplit("input_DIR/",1)[1])
        # print("-image_face_detect--",image_face_detect)
        # print("    "*90)
        
        # root_dir = "../data_dir/face_detected/"
        # face_out_rootDIR= "../data_dir/jungle_images/output_DIR/"
        # cv2.imwrite(face_out_rootDIR+image_face_detect+'__face_.png', image_with_meta_table) # TODO -- save -- root_dir+

    @classmethod
    def iterate_unique_colors(self,
                              unique_colors,
                              image_rgb,
                              result_dict):
        """ 
        """
        for iter_color in unique_colors: # Assign human-readable color names and determine positions
            color_tuple = tuple(iter_color) # Convert the color to a tuple for dictionary lookup
            color_tuple = tuple(int(x) for x in color_tuple) ##required_tuple -- # Convert NumPy tuple to a standard Python tuple of integers
            logger.debug("--get_colors_from_face--AAA--color_tuple-> %s" ,color_tuple)
            #result_dict = {} ## TODO -- Container Position 
            color_name = rgb_color_mapping.get(color_tuple, f"unknown_{color_tuple}") # Get the color name from the mapping
            if "unk" in str(color_name):
                logger.debug("--get_colors_from_face-ccc--UNK-aa-color_name-> %s" ,color_name)
                nearest_rgb, nearest_color = self.get_unknown_colors_rgb(color_tuple, rgb_color_mapping)
                logger.debug("--get_colors_from_face-ccc--UNK--aa--nearest_color-> %s" ,nearest_color)
                if "gray" in str(nearest_color):
                    pass
                else:
                    mask = np.all(image_rgb == iter_color, axis=-1) # Create a mask for the current color
                    position = np.argmax(np.any(mask, axis=0)) # Find the position of the color layer (first occurrence along the x-axis)
                    result_dict[color_name] = int(position)# Map the color name to its position in the result dictionary
                    logger.debug("--get_colors_from_face-mmm-1--result_dict-> %s" ,result_dict)
            else:
                logger.debug("--get_colors_from_face-ccc--AA-color_name-> %s" ,color_name)
                mask = np.all(image_rgb == iter_color, axis=-1) # Create a mask for the current color
                position = np.argmax(np.any(mask, axis=0)) # Find the position of the color layer (first occurrence along the x-axis)
                result_dict[color_name] = int(position)# Map the color name to its position in the result dictionary
                logger.debug("--get_colors_from_face-nnn--11--result_dict-> %s" ,result_dict)

        logger.debug("--get_colors_from_face--nnn-22-result_dict-> %s" ,result_dict)
        # unique_colors = np.unique(gray_image) # Find unique colors in the image
        # result_dict = {} # Initialize the result dictionary
        # for color in unique_colors: # Iterate through the unique colors and determine their positions
        #     mask = np.where(gray_image == color, 1, 0) # Create a mask for the current color
        #     position = np.argmax(np.sum(mask, axis=0) > 0) # Find the position of the color layer
        #     result_dict[color] = position# Map the color to its position in the result dictionary
        # logger.debug("--get_colors_from_face----result_dict-> %s" ,result_dict)
    
    
    @classmethod
    def get_colors_from_face(self,rectangle_portion_only,
                             face_bbox_coords_detection=None):
        """ 
        """
        logger.debug("--get_colors_from_face---HIT---> ") #
        # Get the height and length of the image
        height, length, _ = rectangle_portion_only.shape
        print(f"Image Height (Y-axis): {height} pixels")
        print(f"Image Length (X-axis): {length} pixels")

        gray_image = cv2.cvtColor(rectangle_portion_only, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
        unique_colors = np.unique(gray_image)
        logger.debug("--get_colors_from_face---unique_colors---1-> %s" ,unique_colors)
        
        #pixels = image_rgb.reshape(-1, 3) # Reshape the image to a 2D array of pixels
        #unique_colors = np.unique(pixels, axis=0) # Find unique RGB colors in the image
        #unique_colors = np.unique(rectangle_portion_only, axis=0) # Find unique RGB colors in the image
        image_rgb = cv2.cvtColor(rectangle_portion_only, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB -- Convert from BGR to RGB (OpenCV loads images in BGR format by default)
        logger.debug("--get_colors_from_face---mmm-----image_rgb--> %s" ,type(image_rgb))
        unique_colors_1 = np.unique(image_rgb.reshape(-1, image_rgb.shape[2]), axis=0)

        logger.debug("--get_colors_from_face---unique_colors_1--2nd Occurence---> %s" ,unique_colors_1)
        color_layer_positions = {}
        for iter_color in unique_colors:
            # Create a mask for the current color
            mask = np.where(gray_image == iter_color, 1, 0)
            # Find the start and end positions on the X-axis (length)
            x_start = np.argmax(np.any(mask, axis=0))  # First column with the color
            x_end = length - np.argmax(np.any(mask[:, ::-1], axis=0)) - 1  # Last column with the color
            # Find the start and end positions on the Y-axis (height)
            y_start = np.argmax(np.any(mask, axis=1))  # First row with the color
            y_end = height - np.argmax(np.any(mask[::-1, :], axis=1)) - 1  # Last row with the color
            
            # Store the positions in the dictionary
            color_layer_positions[iter_color] = {
                "x_start": x_start,
                "x_end": x_end,
                "y_start": y_start,
                "y_end": y_end
            }

        logger.debug("--get_colors_from_face---color_layer_positions--> %s" ,color_layer_positions)
        result_dict = {}  # TODO -- Container Position

        #  TODO -##TODO_9thMARCHMARCH--- Ok below -- Dont Run Too many Logs -- 
        # self.iterate_unique_colors(unique_colors_1,
        #                       image_rgb ,
        #                       result_dict)

                    
        ## TODO -- below Code OK -- - NOT REQUIRED - DONE ABOVE 
        # image_rgb = cv2.cvtColor(rectangle_portion_only, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB (OpenCV loads images in BGR format by default)
        # logger.debug("--get_colors_from_face---mmm-----image_rgb--> %s" ,type(image_rgb))
        # #pixels = image_rgb.reshape(-1, 3) # Reshape the image to a 2D array of pixels
        # #unique_colors = np.unique(pixels, axis=0) # Find unique RGB colors in the image
        # #unique_colors = np.unique(rectangle_portion_only, axis=0) # Find unique RGB colors in the image
        # image_rgb = cv2.cvtColor(rectangle_portion_only, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB
        # unique_colors = np.unique(image_rgb.reshape(-1, image_rgb.shape[2]), axis=0)
        # logger.debug("--get_colors_from_face-ccc--unique_colors--> %s" ,type(unique_colors)) ##class 'numpy.ndarray'>
        # logger.debug("--get_colors_from_face-ccc--unique_colors--> %s" ,unique_colors)
        # result_dict = {}  # TODO -- Container Position 

    @classmethod
    def get_unknown_colors_rgb(self,rgb_tuple, rgb_color_mapping):
        """ 
        """
        min_distance = float('inf')
        nearest_color = "unknown"
        nearest_rgb = (0, 0, 0)
        for color_rgb, color_name in rgb_color_mapping.items():
            # Calculate Euclidean distance between the given RGB and the mapped RGB
            distance = np.sqrt(
                (rgb_tuple[0] - color_rgb[0]) ** 2 +
                (rgb_tuple[1] - color_rgb[1]) ** 2 +
                (rgb_tuple[2] - color_rgb[2]) ** 2
            )
        
        # Update nearest color if this distance is smaller
        if distance < min_distance:
            min_distance = distance
            nearest_color = color_name
            nearest_rgb = color_rgb
    
        return nearest_rgb, nearest_color
        # nearest_rgb, nearest_color = find_nearest_color(rgb_tuple, rgb_color_mapping)
        # print(f"Input RGB: {rgb_tuple} -> Nearest RGB: {nearest_rgb}, Color: {nearest_color}")

    @classmethod
    def process_indl_faces(self,face_bbox_coords_detection,rectangle_portion_only): ##ls_faces_coord,image_for_process)
        """ 
        """
        
        ## TODO -- ##TODO_9thMARCHMARCH -- OK below -- Dont Run too Many LOGS 
        #self.get_colors_from_face(rectangle_portion_only,face_bbox_coords_detection)

        # for iter_face in range(len(ls_faces_coord)):  # Iterate through detected faces in current frame
        #     x1, y1, x2, y2 = ls_faces_coord[iter_face]  # Extract box coordinates
        #     logger.debug("-AA-==A--results_face_detect--x1, y1, x2, y2--> %s" ,x1)
        #     face_roi_region = image_for_process[int(y1):int(y2), int(x1):int(x2)]
        # logger.debug("--results_face_detect----face_roi_region-> %s" ,face_roi_region)
        # logger.debug("--results_face_detect----face_roi_region-> %s" ,type(face_roi_region))
        # Draw a red thick border around the ROI

        # Define the color for the border (BGR format: Red = (0, 0, 255))
        # border_color = (0, 0, 255)  # Red color

        # # Define the thickness of the border (in pixels)
        # border_thickness = 10  # Thick border

        # cv2.rectangle(image_for_process, (int(y1), int(y2)), (int(x1), int(x2)), border_color, border_thickness)
        # # Save the result (optional)
        # cv2.imwrite('image_face_roi_border.jpg', image_for_process)

        



            # average_color_region = np.mean(face_roi_region, axis=(0, 1))  # Average over height and width
            # logger.debug("--results_face_detect--TYPE----average_color_region-> %s" ,type(average_color_region))
            # logger.debug("--results_face_detect----average_color_region-> %s" ,average_color_region)
            # # Convert to integers (optional, but often useful for display/OpenCV)
            # color_rgb_int = average_color_region.astype(int)
            # # Print the integer RGB values
            # logger.debug("--results_face_detect----color_rgb_int-> %s" ,color_rgb_int)
            # # If you need a tuple:
            # color_rgb_tuple = tuple(color_rgb_int)
            # #print("RGB tuple:", color_rgb_tuple)
            # # If you want to format it nicely:
            # #print(f"RGB: ({color_rgb_int[0]}, {color_rgb_int[1]}, {color_rgb_int[2]})")

            # # --- Dominant Color Calculation (using k-means) --- k = 3
    
            # k_means_k = 3
            # pixels = face_roi_region.reshape(-1, 3).astype(np.float32)  # Reshape for k-means
            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            # _, labels, centers = cv2.kmeans(pixels, k_means_k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # dominant_colors = centers.astype(int)  # Convert back to integers
            # logger.debug("--results_face_detect--TYPE----dominant_colors-> %s" ,type(dominant_colors))
            # logger.debug("--results_face_detect---dominant_colors-> %s" ,dominant_colors)

            ## TODO -- Convert to COLOR NAMES 
            ## https://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green-with-python

        # for result in results_face_detect: 
        #     boxes = result.boxes  # Get the bounding boxes
        #     xyxy = boxes.xyxy.astype(int)  # Convert box coordinates to integers (important for OpenCV)
        #     conf = boxes.conf  # Get confidence scores
        #     cls = boxes.cls  # Get class labels (usually 0 for face detection)

    
    @classmethod
    def table_on_image_frame(self,annotated_frame):
        """ 
        """
        # Load an image
        #image = cv2.imread("input_image.jpg")  # Replace with your image path
        image = annotated_frame
        # Define the table content
        table_data = [
                    "PERSON TYPE",
                    "HEADGEAR TYPE"
                ]

        # Define table properties
        table_width = 100  # Width of the table
        table_height = 70  # Height of the table
        font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
        font_scale = 0.3  # Font size
        font_color = (255, 255, 255)  # White text color
        line_type = cv2.LINE_AA  # Anti-aliased line type
        background_color = (0, 100, 50)  # Dark green background (BGR format)

        # Create a dark green background for the table
        table_background = np.zeros((table_height, table_width, 3), dtype=np.uint8)
        table_background[:] = background_color

        # Add text to the table
        y_offset = 10  # Vertical offset for the first row
        for i, row in enumerate(table_data):
            text_size = cv2.getTextSize(row, font, font_scale, 1)[0]
            text_x = (table_width - text_size[0]) // 2  # Center text horizontally
            text_y = y_offset + (i * 30)  # Adjust vertical position for each row
            cv2.putText(table_background, row, (text_x, text_y), font, font_scale, font_color, 1, line_type)

            # Define the position of the table (top-right corner)
            table_x = image.shape[1] - table_width  # X-coordinate (right side)
            #table_y =  0 # Y-coordinate== TOP of IMAGE 
            table_y =  image.shape[0] - table_height # Y-coordinate==BOTTOM of IMAGE (top side)
            # Overlay the table on the image
            image[table_y:table_y + table_height, table_x:table_x + table_width] = table_background

            # Save or display the result
            #cv2.imwrite("output_TABLE_image.jpg", image)  # Save the output image
            # cv2.imshow("Image with Table", image)  # Display the image
            # cv2.waitKey(0)  # Wait for a key press
            # cv2.destroyAllWindows()  # Close all OpenCV windows

        return image

class ObjDetHFRtDetr:
    """
    """
    @classmethod
    def object_detect_RT_DETR(self,image_frame_path):
        """ 
        """
        image_local = Image.open(image_frame_path) #
        ls_results = pipeline_rtdetr_v2(image_frame_path, threshold=0.3)
        print("---type(res)---------",type(ls_results))
        print("   "*100)
        self.draw_PIL_Image(ls_results,
                            image_local,
                            image_frame_path)
        return ls_results
    
    @classmethod
    def draw_PIL_Image(self,ls_results,
                       image_local,
                       image_frame_path):
        """ 
        """
        from PIL import ImageDraw
        LS_COLORS = ["green","yellow","red","blue","green","yellow","red","blue","green","yellow","red","blue","green","yellow","red","blue","green","yellow","red","blue","green","yellow","red","yellow","red","blue","green","yellow","red","blue","green","yellow","red","blue","green","yellow","red"]
        # LS_COLORS = [
        #                 [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
        #                 [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
        #             ] * 100
        
        # colors for visualization
        # COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
        #         [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

        annotated_image = image_local.copy()
        #draw = ImageDraw.Draw(annotated_image)
        plt.figure(figsize=(25,25))
        plt.imshow(annotated_image)
        ax = plt.gca()
        try:
            for iter_k in range(len(ls_results)): ## Original -- #for i, result in enumerate(ls_results):
                dict_res_1 = ls_results[iter_k]#["box"]
                print(dict_res_1)
                print("   "*100) ##{'score': 0.31572669744491577, 'label': 'cup', 'box': {'xmin': 407, 'ymin': 498, 'xmax': 440, 'ymax': 620}}
                dict_box = dict_res_1["box"]
                print(dict_box)
                #color = tuple([int(x * 255) for x in LS_COLORS[iter_k]])
                xmin, ymin, xmax, ymax = dict_box["xmin"], dict_box["ymin"], dict_box["xmax"], dict_box["ymax"]
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                            fill=False, color=LS_COLORS[iter_k], linewidth=4)) # Drawing_BBox
                ax.text(xmin, ymin,dict_res_1["label"],fontsize=25,
                        bbox=dict(facecolor='yellow', alpha=0.5)) # Drawing_LABEL
                plt.axis('off')
                root_save_dir = "../data_dir/bbox_image_saved/--TODO/"
                image_named_bbox = str(str(image_frame_path).rsplit("input_DIR/",1)[1])
                print("---image_named_bbox----",image_named_bbox)
                print("- -OK-----   "*10)
                plt.savefig(root_save_dir+str(image_named_bbox)+".png", bbox_inches='tight') ## TODO -- save
        except Exception as err:
            print(err)
            pass

        #     draw.rectangle((xmin, ymin, xmax, ymax), fill=None, outline=color, width=7)
        #     draw.text((xmin, ymin, xmax, ymax), text=dict_res_1["label"],fill="yellow",align ="left")#font=10)
        # annotated_image.show()
        # annotated_image.save("anno_img_ImageDraw_1.png")


#------------#------------#------------#------------#------------#------------
# read original 
# img = cv2.imread('original.png')

# # call our function
# img_zoomed_and_cropped = zoom_center(img)

# # write zoomed and cropped version
# cv2.imwrite('zoomed_and_cropped.png', img_zoomed_and_cropped)


######---------------
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)



######-------# ORIGINAL CODE --------
# inputs = processor(images=image, return_tensors="pt") ## RESNET--101
# outputs = model(**inputs) ## RESNET--101

#print(f"---Model-outputs->> {outputs}")
#DetrObjectDetectionOutput.logits
#outputs.logits
#print(f"---Model-outputs->> {outputs.logits}") ## Original Code had -- pred_logits
# keep only predictions with 0.7+ confidence
#probas = outputs['pred_logits'].softmax(-1)[0, :, :-1] ## ORIGINAL CODE 