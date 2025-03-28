from util_logger import setup_logger_linux
logger = setup_logger_linux(module_name=str(__name__))

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections


class LoadLocalModels:
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