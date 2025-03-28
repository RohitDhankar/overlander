import cv2
import numpy as np

def get_dominant_color_above_bbox(image, bbox, range_region_cm, range_region_1_cm, k=5): # k = number of dominant colors to consider
    """
    Gets the dominant RGB color of a region above a bounding box using k-means clustering.

    Args:
        image: The input image (NumPy array).
        bbox: The bounding box coordinates [x1, y1, x2, y2] (NumPy array or list).
        range_region_cm: The height of the first region above the bbox in centimeters.
        range_region_1_cm: The height of the second region above the bbox in centimeters.
        k: The number of dominant colors to consider (default is 5).

    Returns:
        A tuple containing the dominant RGB colors of the two regions, or None if there's an issue.
    """

    x1, y1, x2, y2 = bbox.astype(int)
    height, width, _ = image.shape

    dpi = 72  # Adjust if needed
    cm_to_pixels = dpi / 2.54

    range_region_pixels = int(round(range_region_cm * cm_to_pixels))
    range_region_1_pixels = int(round(range_region_1_cm * cm_to_pixels))

    roi_y1_region = max(0, y1 - range_region_pixels)
    roi_y2_region = y1
    roi_y1_region_1 = max(0, y1 - range_region_pixels - range_region_1_pixels)
    roi_y2_region_1 = y1 - range_region_pixels

    if roi_y1_region < 0 or roi_y2_region > height or roi_y1_region_1 < 0 or roi_y2_region_1 > height:
        print("Warning: ROI is outside the image boundaries. Returning None.")
        return None

    roi_region = image[roi_y1_region:roi_y2_region, x1:x2]
    roi_region_1 = image[roi_y1_region_1:roi_y2_region_1, x1:x2]

    if roi_region.size == 0 or roi_region_1.size == 0:
        print("Warning: ROI is empty. Returning None.")
        return None

    # --- Dominant Color Calculation (using k-means) ---
    def get_k_dominant_colors(roi, k):
        pixels = roi.reshape(-1, 3).astype(np.float32)  # Reshape for k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_colors = centers.astype(int)  # Convert back to integers
        return dominant_colors

    dominant_colors_region = get_k_dominant_colors(roi_region, k)
    dominant_colors_region_1 = get_k_dominant_colors(roi_region_1, k)


    return dominant_colors_region, dominant_colors_region_1

# ... (rest of the code is the same as the previous example, but use the new function)

image_path = "path/to/your/image.jpg"
image = cv2.imread(image_path)
BBOX_OF_INTEREST = np.array([653.07, 632.32, 830.79, 768])
RANGE_REGION = 2
RANGE_REGION_1 = 4
k = 3 # Consider top 3 dominant colors

dominant_colors = get_dominant_color_above_bbox(image, BBOX_OF_INTEREST, RANGE_REGION, RANGE_REGION_1, k)

if dominant_colors is not None:
    dominant_colors_region, dominant_colors_region_1 = dominant_colors
    print("Dominant colors of region above bbox:", dominant_colors_region)
    print("Dominant colors of region_1 above bbox:", dominant_colors_region_1)

    # ... (visualization code remains the same)

else:
    print("Could not get colors.")