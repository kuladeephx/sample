from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture
import base64
import requests

runner = Runner(
    which=Which.VisionPlain(
        model_id="microsoft/Phi-3.5-vision-instruct",
        arch=VisionArchitecture.Phi3V,
    ),
)

image_url = 'https://image.lexica.art/full_jpg/0000463a-7eaa-4fa9-96d8-1f41f3e8dca4'
image_url_2 ='https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'

encoded_string = base64.b64encode(requests.get(image_url).content).decode("utf-8")
encoded_string_2 = base64.b64encode(requests.get(image_url_2).content).decode("utf-8")
res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="phi3v",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<|image_1|>\n<|image_2|>\nAre these two images same? Just answer Yes or No",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str(encoded_string)
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str(encoded_string_2),
                        },
                    },
                    
                ],
            }
        ],
        max_tokens=256,
        
        temperature=0.0,
    )
)
