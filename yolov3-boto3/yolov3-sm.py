from doctest import REPORT_CDIFF
from email.mime import base
import boto3
import cv2
import base64
from io import BytesIO
import time 
import datetime

client = boto3.client('sagemaker-runtime', region_name='ap-south-1')

if __name__ == "__main__":
    def deserialize(obj):
        """Convert JSON dicts back into objects."""
        # Be careful of shallow copy here
        target = dict(obj)
        class_name = None
        if "__class__" in target:
            class_name = target.pop("__class__")
        if "__module__" in obj:
            obj.pop("__module__")
        # Use getattr(module, class_name) for custom types if needed
        if class_name == "datetime":
            return datetime(tzinfo=utc, **target)
        if class_name == "StreamingBody":
            return StringIO(target["body"])
        # Return unrecognized structures as-is
        return obj 
    #  read the image
    im = cv2.imread("./test_image.jpg")
    im_resize = cv2.resize(im, (500, 500))
    is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)
    byte_im = im_buf_arr.tobytes()
    # _,im_encoded = cv2.imencode('.JPEG', img)
    # im_btye = base64.b16encode(im_encoded)
    # print(type(im_btye))
    t1 = time.time()
    response = client.invoke_endpoint(
    EndpointName='Endpoint-GluonCV-YOLOv3-Object-Detector-1',
    Body=byte_im,
    ContentType='image/jpeg',
    CustomAttributes='{"threshold": 0.2}')
    t2 = time.time()
    delta = t2 - t1
    print("inference: ", delta)
    print(response)
