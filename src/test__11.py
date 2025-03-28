from itertools import zip_longest
import cv2
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class FaceProcessor:
    def __init__(self):
        pass

    def add_race_gender_labels(self, image, gender, race, x, y):
        """
        Add gender and race labels to the image at the specified coordinates.
        """
        label = f"{gender} | {race}"  # Combine gender and race into a single label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (0, 255, 0)  # Green color in BGR format

        # Calculate text position (slightly above the top-left corner of the bounding box)
        text_x = x
        text_y = y - 10  # 10 pixels above the bounding box

        # Ensure the text does not go out of the image bounds
        if text_y < 0:
            text_y = y + 20  # Move the text below the bounding box if it goes out of bounds

        # Add the text to the image
        cv2.putText(image, label, (text_x, text_y), font, font_scale, text_color, font_thickness)

    def draw_bbox_opencv(self, results_extract_faces, 
                         ls_dominant_race, ls_dominant_gender, 
                         image_for_bbox, face_out_rootDIR, face_image_name):
        """
        Draw bounding boxes and add labels for gender and race on the detected faces.
        """
        # Define bounding box color and thickness
        color = (0, 255, 0)  # Green color in BGR format
        thickness = 2  # Thickness of the bounding box

        # Iterate over the detected faces and their corresponding gender/race labels
        for iter_f, (tag_race, tag_gender) in enumerate(zip_longest(ls_dominant_race, ls_dominant_gender, fillvalue="Unknown")):
            # Get the facial area coordinates
            dict_1 = results_extract_faces[iter_f]
            dict_facial_area = dict_1.get("facial_area", None)

            if dict_facial_area is None:
                logger.warning(f"No facial area found for face {iter_f}")
                continue

            logger.debug("--draw_bbox_opencv---dict_facial_area---> %s", dict_facial_area)

            # Extract bounding box coordinates
            bbox_x = dict_facial_area["x"]
            bbox_y = dict_facial_area["y"]
            bbox_width = dict_facial_area["w"]
            bbox_height = dict_facial_area["h"]

            # Draw the bounding box
            cv2.rectangle(image_for_bbox, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), color, thickness)

            # Add gender and race labels
            self.add_race_gender_labels(image_for_bbox, tag_gender, tag_race, bbox_x, bbox_y)

        # Save the image with bounding boxes and labels
        output_path = f"{face_out_rootDIR}{face_image_name}_1_.png"
        cv2.imwrite(output_path, image_for_bbox)
        logger.info(f"Image with bounding boxes saved to: {output_path}")