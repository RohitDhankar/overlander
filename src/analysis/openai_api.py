
import os
import openai
import base64
from openai import OpenAI


openai.api_key = os.environ["OPENAI_API_KEY"]
#print(openai.api_key) # OK Dont 

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
#image_rootDIR= "../data_dir/weapon_imgs/weapon_1.png"
#image_path = "/home/dhankar/temp/01_25/ipWebCam/data_dir/weapon_imgs/weapon_1.png"
#weapon_image = "/home/dhankar/temp/01_25/ipWebCam/data_dir/weapon_imgs/2_ak.jpeg"
# ```python
# {"WEAPONS":"yes",
#  "TOTAL_WEAPONS_COUNT":1,
#  "WEAPONS_TYPE_AND_COUNT":{"AK-47":1}
# }
# ```

weapon_image = "/home/dhankar/temp/01_25/ipWebCam/data_dir/weapon_imgs/5_2.jpeg"

# FACE Image 
#face_image = "/home/dhankar/temp/01_25/ipWebCam/data_dir/weapon_imgs/hostes_2.png"
##{"FACES":"yes","FACES_COUNT":3,"FACES_TYPE":{"MALE":1,"FEMALE":2}}
#
#face_image = "/home/dhankar/temp/01_25/ipWebCam/data_dir/weapon_imgs/many_faces.png"
# Sorry
#
face_image = "/home/dhankar/temp/01_25/ipWebCam/data_dir/weapon_imgs/hostess_1.png"


# Getting the Base64 string
base64_image = encode_image(weapon_image)

prompt_weapon_image = """Is there any kind of weapon or rifle in this image? "

                "If yes you need to respond in this manner with a Python Dictionary -
                        {"WEAPONS":"yes",
                        "TOTAL_WEAPONS_COUNT":7,
                        "WEAPONS_TYPE_AND_COUNT":{"AK-47":2,
                                                "SHOTGUN":1,
                                                "ROCKET_RPG":1,
                                                "GRENADE":1,
                                                ANY_OTHER_WEAPON:2},
                        }
                  , "
                "if you DO NOT IDENTIFY any WEAPON only respond with - NO_WEAPON 
                Do not respond with any other TEXT as a response"""


prompt_face_image = """ Is there any Human Face in the Image ?

1. If there is a FACE - count the Number of FACES in the IMAGE.

1.a) You need to respond in this manner with a Python Dictionary - 

        {"FACES":"yes",
        "FACES_COUNT":2,
        "FACES_TYPE":{"MALE":2,
        "FEMALE":1},
        }

2. Do Not respond with any other text 

"""

completion = client.chat.completions.create(
    #model="gpt-4o", # YES 
    #model="o3-mini", # NO 
    ##gpt-4o-mini 
    model="gpt-4o-mini", # YES - OK -- AK47
    messages=[
        {
            "role": "user",
            "content": [
                { "type": "text", "text": prompt_weapon_image }, ##prompt_face_image+
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high",
                    },
                },
            ],
        }
    ],
)
print(completion)

print(completion.choices[0].message.content)




# # import os
# # from openai import OpenAI

# # client = OpenAI(
# #     # This is the default and can be omitted
# #     api_key=os.environ.get("OPENAI_API_KEY"),
# # )

# # response = client.responses.create(
# #     model="gpt-4o",
# #     instructions="You are a coding assistant that talks like a pirate.",
# #     input="How do I check if a Python object is an instance of a class?",
# # )

# # print(response.output_text)