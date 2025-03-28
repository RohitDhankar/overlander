# import os 
# from azure.core.credentials import AzureKeyCredential
# from azure.ai.vision.face import FaceClient
# from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel

import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


def check_env_variables():
    """
    # Set the values of your computer vision endpoint and computer vision key
    # as environment variables:
    """
    try:
        endpoint = os.environ["VISION_ENDPOINT"]
        key = os.environ["VISION_KEY"]
        print("OK --environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        print("  -ok--        "*4)

        return endpoint , key
    except KeyError:
        print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        print("Set them before running this sample.")
        exit()


endpoint , key = check_env_variables()
# Create an Image Analysis client
az_vision_client = ImageAnalysisClient(
                    endpoint=endpoint,
                    credential=AzureKeyCredential(key)
                )
print(" --[INFO]--az_vision_client-->>",type(az_vision_client))
## <class 'azure.ai.vision.imageanalysis._patch.ImageAnalysisClient'>

### VisualFeatures.CAPTION--> 
visual_features =[
        VisualFeatures.TAGS,
        VisualFeatures.OBJECTS,
        VisualFeatures.CAPTION,
        VisualFeatures.DENSE_CAPTIONS,
        VisualFeatures.READ,
        VisualFeatures.SMART_CROPS,
        VisualFeatures.PEOPLE,
    ]



from PIL import Image as PIL_IMAGE
import io
#img_byte_arr = io.BytesIO()

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

def get_captions(image_1):
    """ 
    """
    img_byte_arr = io.BytesIO()
    dict_captions = {}
    # Get a caption for the image. This will be a synchronously (blocking) call.
    # result_analyze_from_url = az_vision_client.analyze_from_url(
    #             image_url="https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png",
    #             visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
    #             gender_neutral_caption=True,  # Optional (default is False)
    #         )
    # print("---Type-result_analyze_from_url---->",type(result_analyze_from_url))
    # print("---Type-result_analyze_from_url---->",result_analyze_from_url)


    #static_dir = "static/image_uploads/"
    #ls_image_files = ["static/image_uploads/2_ak.jpeg","static/image_uploads/1_ak.jpeg"]
    #ls_files_uploads = get_static_local_list()

    #for image_file in range(len(ls_files_uploads)):
    #image_1 = str(ls_files_uploads[image_file]) ##static_dir + 
        #print("---Type----",type(image_1))
    print("---Type--PATH--",image_1)
    
    image_2 = PIL_IMAGE.open(image_1)
    image_2.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    print("--[INFO]---image_3----->>",type(img_byte_arr))

    result_local_file = az_vision_client.analyze(         
                                image_data=img_byte_arr, 
                                visual_features=[VisualFeatures.CAPTION,
                                                                    VisualFeatures.READ]
                            )

    print(" Caption:----------------->>")
    if result_local_file.caption is not None:
        #print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}")
        #print(f"   '{result_local_file.caption.text}', Confidence {result_local_file.caption.confidence:.4f}")
        dict_captions["image_auto_captioned"] = str(result_local_file.caption.text)

    if result_local_file.read is not None:
        #print(" =result_local_file.read- ",result_local_file.read)
        try:
            for line in result_local_file.read.blocks[0].lines:
                #print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
                for word in line.words:
                    #print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")
                    dict_captions["image_text_found"] = str(word.text)
        except Exception as err:
            print(err)
            pass

    print(" =EOF==   "*2)
    return dict_captions