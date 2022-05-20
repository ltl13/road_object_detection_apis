import os, shutil
from werkzeug.exceptions import BadRequest
from flask import Flask, render_template, request, make_response
import torch
import cv2
from PIL import Image
import io
import sys
sys.path.append('../yolov5')

from detect import detectImg
app = Flask(__name__)


def get_detection(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    file_path = '../images/cache.jpg'
    img_src = img.save(file_path)
    result = detectImg(weights='../weights/best.pt', source='../images/cache.jpg',
                       conf_thres=0.5, iou_thres=0.5, imgsz=(736, 736))

    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

    return result

@app.route('/', methods=['POST'])
def predict():
    file = extract_img(request)
    img_bytes = file.read()
    results = get_detection(img_bytes)


    RGB_img = cv2.cvtColor(results, cv2.COLOR_BGR2RGB)
    im_arr = cv2.imencode('.jpg', RGB_img)[1]
    response = make_response(im_arr.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'
    return response


def extract_img(request):
    # checking if image uploaded is valid
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")
    file = request.files['file']
    if file.filename == '':
        raise BadRequest("Given file is invalid")
    return file

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
