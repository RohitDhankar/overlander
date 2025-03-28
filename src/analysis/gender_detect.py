

from util_logger import setup_logger_linux
logger = setup_logger_linux(module_name=str(__name__))
import os , cv2 

# from keras.models import load_model
# from mtcnn.mtcnn import MTCNN
# # Load pre-trained models
# gender_model = load_model('xception_gender_model.h5')
# detector = MTCNN()# Function to detect gender

from deepface import DeepFace as native_deepface

class DeepfaceDetect:
    """ 
    """

    def add_race_gender_labels(self,image_for_bbox,tag_race,bbox_x,bbox_y):
        """ 
        """
        # Add a label on the top-left edge of the bounding box
        label = tag_race  # Text to display
        font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
        font_scale = 0.8  # Font scale
        font_thickness = 2  # Thickness of the font
        text_color = (0, 255, 255)  # Text color (green in BGR format)
        # Calculate the position for the label (slightly above the top-left corner of the bounding box)
        text_x = bbox_x # Align with the left edge of the bounding box
        text_y = bbox_y - 10  # Place the text 10 pixels above the top edge of the bounding box
        # Ensure the text does not go out of the image bounds
        if text_y < 0:
            text_y = bbox_y + 20  # Move the text below the bounding box if it goes out of bounds
        # Add the text to the image
        cv2.putText(image_for_bbox, label, (text_x, text_y), font, font_scale, text_color, font_thickness)

    def draw_bbox_opencv(self,
                         image_local_path,
                         results_extract_faces,
                         ls_dominant_race,
                         ls_dominant_gender):
        """ 
        # Draw a bounding box around the face = Parameters: image, top-left corner (x, y), bottom-right corner (x + w, y + h), color (BGR), thickness
        """
        face_out_rootDIR= "../data_dir/deepface/output_dir/"
        if not os.path.exists(face_out_rootDIR):
            os.makedirs(face_out_rootDIR)
        face_image_name = str(image_local_path).rsplit("/",1)[1]
        logger.debug("--draw_bbox_opencv---face_image_name---> %s" ,face_image_name)
        print("--split-----",face_image_name) # TODO -- 

        color = (0, 255, 0)  # Green color in BGR format
        thickness = 2  # Thickness of the bounding box
        image_for_bbox = cv2.imread(image_local_path) ##
        if len(results_extract_faces) == len(ls_dominant_race):
            logger.debug("--draw_bbox_opencv---SAME---len(ls_dominant_race)--> %s" ,len(ls_dominant_race))
            #
            for iter_f in range(len(results_extract_faces)):
                tag_race = ls_dominant_race[iter_f]
                tag_gender = ls_dominant_gender[iter_f]
                tag_gender_race = str(tag_race + "__"+tag_gender)
                dict_1 = results_extract_faces[iter_f]
                dict_facial_area = dict_1.get("facial_area",None)
                
                logger.debug("--draw_bbox_opencv---dict_facial_area---> %s" ,dict_facial_area)
            
                bbox_x = dict_facial_area["x"]
                bbox_y = dict_facial_area["y"]
                bbox_width = dict_facial_area["w"]
                bbox_height = dict_facial_area["h"]
                cv2.rectangle(image_for_bbox, (bbox_x, bbox_y),(bbox_x + bbox_width, bbox_y + bbox_height), color, thickness)
                self.add_race_gender_labels(image_for_bbox,tag_gender,bbox_x,bbox_y)
                cv2.imwrite(face_out_rootDIR+str(face_image_name)+"_2_.png", image_for_bbox)

        else:
            logger.debug("--draw_bbox_opencv--NOT-SAME-aa--len(ls_dominant_race)--> %s" ,len(ls_dominant_race))
            logger.debug("--draw_bbox_opencv--NOT-SAME-aa--len(results_extract_faces)--> %s" ,len(results_extract_faces))
            try:
                from itertools import zip_longest

                #for iter_f in range(len(results_extract_faces)):
                for iter_f, (tag_race, tag_gender) in enumerate(zip_longest(ls_dominant_race, ls_dominant_gender, fillvalue="Unknown")):
                    # tag_race = ls_dominant_race[iter_f]
                    # tag_gender = ls_dominant_gender[iter_f]
                    #tag_gender_race = str(tag_race + "__"+tag_gender)

                    dict_1 = results_extract_faces[iter_f]
                    dict_facial_area = dict_1.get("facial_area",None)
                    logger.debug("--draw_bbox_opencv---dict_facial_area---> %s" ,dict_facial_area)
                
                    bbox_x = dict_facial_area["x"]
                    bbox_y = dict_facial_area["y"]
                    bbox_width = dict_facial_area["w"]
                    bbox_height = dict_facial_area["h"]
                    cv2.rectangle(image_for_bbox, (bbox_x, bbox_y),(bbox_x + bbox_width, bbox_y + bbox_height), color, thickness)
                    #self.add_race_gender_labels(image_for_bbox,tag_gender,bbox_x,bbox_y)
                    cv2.imwrite(face_out_rootDIR+str(face_image_name)+"_2_.png", image_for_bbox)
            except Exception as err:
                print(err)
                pass


    def extract_faces(self,image_local_path):
        """ 
        """
        ls_dominant_race = []
        ls_dominant_gender = []
        logger.debug("--extract_faces---image_local_path--aa---> %s" ,image_local_path)
        detectors = ["mtcnn"] ## "retinaface", -- ERRORS with -- "retinaface",
        gender_objs_deepface = self.get_gender_obj(image_local_path)
        logger.debug("--extract_faces---gender_objs_deepface--> %s" ,type(gender_objs_deepface))
        logger.debug("--extract_faces---gender_objs_deepface-LEN---aa--> %s" ,len(gender_objs_deepface))
        logger.debug("--extract_faces---gender_objs_deepface--> %s" ,gender_objs_deepface)

        for iter_face in range(len(gender_objs_deepface)):
            dict_1 = gender_objs_deepface[iter_face]
            dominant_gender = dict_1.get("dominant_gender",None)
            logger.debug("--extract_faces---dominant_gender--> %s" ,dominant_gender)
            ls_dominant_gender.append(dominant_gender)
            
            dominant_race = dict_1.get("dominant_race",None)
            logger.debug("--extract_faces---dominant_race--> %s" ,dominant_race)
            ls_dominant_race.append(dominant_race)

        logger.debug("--extract_faces--LEN-ls_dominant_gender--> %s" ,len(ls_dominant_gender))
        logger.debug("--extract_faces---ls_dominant_gender--> %s" ,ls_dominant_gender)
        #
        logger.debug("--extract_faces--LEN-ls_dominant_race--> %s" ,len(ls_dominant_race))
        logger.debug("--extract_faces---ls_dominant_race--> %s" ,ls_dominant_race)


        for idx, detector_backend in enumerate(detectors):
            results_extract_faces = native_deepface.extract_faces(img_path=image_local_path, 
                                                                  detector_backend=detector_backend,
                                                                  enforce_detection=False)
            logger.debug("--extract_faces---results_extract_faces-TYPE--> %s" ,type(results_extract_faces))
            logger.debug("--extract_faces---results_extract_faces---> %s" ,results_extract_faces)
            self.draw_bbox_opencv(image_local_path,
                                  results_extract_faces,
                                  ls_dominant_race,
                                  ls_dominant_gender)


    def get_gender_obj(self,image_local_path):
        """ 
        """
        objs_deepface = native_deepface.analyze(
            img_path = image_local_path,
            #actions = ['age', 'gender', 'race', 'emotion'], #TODO # Dont EMOTION -- Compute heavy 
            actions = ['age', 'gender', 'race'],
            enforce_detection=False,
            )
        return objs_deepface
        


# class XceptionFaceClass:
    """ 
    """


    @classmethod
    def detect_gender(self,image_path):
        image = cv2.imread(image_path)
        faces = detector.detect_faces(image)
        for face in faces:
            x, y, width, height = face['box']
            face_image = image[y:y+height, x:x+width]
            face_image = cv2.resize(face_image, (224, 224)) # Resize for model input
            prediction = gender_model.predict(face_image.reshape(1, 224, 224, 3))
            # gender = 'Male' if prediction[^0][^0] > 0.5 else 'Female'
            # # Draw rectangle and label
            # color = (255, 0, 0) if gender == 'Male' else (0, 0, 255)
            # cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
            # cv2.putText(image, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)