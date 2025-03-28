from transformers import (AutoImageProcessor,
                            AutoModelForObjectDetection)
import torch
from transformers import pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

from util_logger import setup_logger_linux
logger = setup_logger_linux(module_name=str(__name__))
import torch , os , cv2
from PIL import Image
import requests ,math
import matplotlib.pyplot as plt ##Error--plot_results-
import time 

checkpoint = "PekingU/rtdetr_v2_r50vd"
# Alternative checkpoints:
# checkpoint = "PekingU/rtdetr_v2_r18vd"
# checkpoint = "PekingU/rtdetr_v2_r34vd"
# checkpoint = "PekingU/rtdetr_v2_r101vd"

image_processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForObjectDetection.from_pretrained(checkpoint).to(device)

COLORS_HFRtDetr_v2_model = [
    [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
] * 100

class AutoModelRtDetrV2:
    """ 
    """
    
    @classmethod
    def obj_detect_HFRtDetr_v2_model(self,image_local_path):
        """ 
        Desc:
            - HFRtDetr_v2 - the Non Pipeline - direct Model call 
        """
        try:
            image_local_frame = Image.open(image_local_path).convert("RGB") #TODO --only in case of PNG Images 
            logger.debug("--model_obj_detection--Type-image_local_frame-> %s" ,type(image_local_frame))
            inputs = image_processor(image_local_frame, return_tensors="pt")
            inputs = inputs.to(device)

            print(inputs.keys())
            logger.debug("--model_obj_detection---inputs.keys()---> %s" ,inputs.keys())
            with torch.no_grad():
                outputs = model(**inputs)
                # postprocess model outputs
                postprocessed_outputs = image_processor.post_process_object_detection(
                                                        outputs,
                                                        target_sizes=[(image_local_frame.height, image_local_frame.width)],
                                                        threshold=0.3,
                                                    )
                
                image_detections = postprocessed_outputs[0]  # take only first image results
                logger.debug("--model_obj_detection--image_detections--> %s" ,image_detections)
                return image_detections , image_local_frame
        except Exception as err:
            logger.error("-ERROR --HFRtDetr_v2----> %s" ,err)

    @classmethod
    def plot_results_HFRtDetr_v2_model(self,image_detections,image_local_frame,image_local_path):
        """ 
        """
        scores = image_detections['scores'].tolist()
        labels = image_detections['labels'].tolist()
        boxes = image_detections['boxes'].tolist()
        plt.figure(figsize=(25,25))
        plt.imshow(image_local_frame)
        ax = plt.gca()
        for score, label, box, color in zip(scores, labels, boxes, COLORS_HFRtDetr_v2_model):
            xmin, ymin, xmax, ymax = box
            ax.add_patch(
                        plt.Rectangle(
                            (xmin, ymin), xmax - xmin, ymax - ymin,
                            fill=False,
                            color=color,
                            linewidth=4,
                        )
                    )
            text = f"{model.config.id2label[label]}: {score:0.2f}"
            ax.text(
                xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.25),
            )
        plt.axis('off')
        root_save_dir = "../data_dir/jungle_images/output_DIR/"
        image_named_bbox = str(str(image_local_path).rsplit("input_DIR/",1)[1])
        print("---image_named_bbox----",image_named_bbox)
        print("- -OK-   "*10)
        plt.savefig(root_save_dir+str(image_named_bbox)+".png", bbox_inches='tight') ## TODO -- save
        #plt.show()



# class PipeLineRtDetrV2:
#     """ 
#     """

#     # checkpoint = "PekingU/rtdetr_v2_r50vd"
#     # pipeline_rtdetr_v2 = pipeline("object-detection", model=checkpoint, image_processor=checkpoint)

# #DetrFeatureExtractor) ##DetrFeatureExtractor ## Deprecated warning 
         