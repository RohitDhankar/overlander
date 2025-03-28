
# Use a pipeline as a high-level helper
from PIL import Image
from transformers import pipeline ##TypeError: 'ImageClassificationPipeline' object is not iterable
import os 

def get_static_local_list():
    """ 
    """
    root_dir = "static/image_uploads/"
    ls_files_uploads = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            ls_files_uploads.append(os.path.join(root, filename))
        # for dirname in dirs:
        #     doSomethingWithDir(os.path.join(root, dirname))
    print("---===ls_files_uploads===-",ls_files_uploads)
    print("  --+  "*10)
    return ls_files_uploads

# root_dir = "static/image_uploads/"
# image_path = root_dir + "arty_1b.jpeg"
# ls_all_images = 

ls_files_uploads = get_static_local_list() 
 
print(f"-Images Scored with Huggingface Transformers-")
print(f"-nsfw ---Not Suitable for WORK-")
print(f"-Model---Falconsai/nsfw_image_detection-")
print("   "*100)


for iter_img in range(len(ls_files_uploads)):

    img = Image.open(ls_files_uploads[iter_img])
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
    #nsfw_classifier(img)
    nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
    nsfw_score_normal_0 = [x["score"] for x in nsfw_classifier(img) if x["label"] != "nsfw"][0]
    nsfw_score_normal = [x["score"] for x in nsfw_classifier(img) if x["label"] != "nsfw"]
    nsfw_label = [x["label"] for x in nsfw_classifier(img) if x["label"] != "nsfw"]#[0]
    all_labels = [x["label"] for x in nsfw_classifier(img)]#[0]
    all_scores = [x["score"] for x in nsfw_classifier(img)]#[0]

    print("   "*100)
    print(f"---Image Name -/ Path--->> {iter_img}")
    print(f"---all_labels--->> {all_labels}")
    print(f"---all_scores--->> {all_scores}")

# print(nsfw_score)

# print(nsfw_score_normal_0)
# print(nsfw_score_normal)

# print(nsfw_label)

# for ele in list(classifier):
#     print(ele)

# print("---type--classifier--->",type(classifier))
# print("---type--classifier--->",classifier)