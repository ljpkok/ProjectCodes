from flask import Flask, jsonify
import config
from flask import request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from inference import inference
from ResNLSTM.lrcn_model import ConvLstm
import torch
import os

app = Flask(__name__)
app.config.from_object(config)
cors = CORS(app)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app.config['CORS_HEADERS'] = 'Content-Type'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'mpg', 'mpeg', 'wmv', 'flv', '3gp', 'm4v', 'mts', 'm2ts', 'ts', 'rm',
                      'rmvb', 'm4a', 'aac', 'mj2', 'mjp2', 'mjpeg', 'jpeg', 'jpg', 'png', 'gif'}

print('Loading model...')
# YOLOv5
# model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
modelYOLO = torch.hub.load('ultralytics/yolov5', 'custom', path='checkpoint/yolov5m.pt')
modelYOLO = modelYOLO.to(device)

# ResLSTM
model = ConvLstm(512, 256, 2, True, 101)
model = model.to(device)
checkpoint = torch.load(os.path.join('checkpoint', 'epoch_100.pth.tar'), map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
print('Model loaded.')


# 判断文件夹后缀
def allow_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello World!'


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return "None"
    else:
        anchor = request.files
        file = request.files.get('files')
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("Start Inference ...")
        cl, obj = inference(filename, device, model, modelYOLO)
        return jsonify({"class": cl, "obj": obj})

if __name__ == "__main__":
    app.run()
