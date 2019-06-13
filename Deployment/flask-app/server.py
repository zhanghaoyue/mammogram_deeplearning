import os.path
import sys
import flask
from flask_uploads import UploadSet, IMAGES, configure_uploads, ALL
from flask import Flask, render_template, request, url_for
from PIL import Image
import time
import cv2
import numpy as np
import config
from werkzeug.utils import secure_filename
from model import Pytorchmodel


zichen_model_dir = os.path.join(os.getcwd(), r'code_zichen')
sys.path.insert(0, zichen_model_dir)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
print(sys.path)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config.from_object(config)
photos = UploadSet('PHOTO')
configure_uploads(app, photos)

segmentation_model_path = './code_zichen/checkpoint/segmentation_model.pth'
classification_model_path = './code_zichen/checkpoint/classification_model.pth'
model = Pytorchmodel(segmentation_model_path, classification_model_path)


@app.route('/upload_image', methods=['POST', 'GET'])
def upload():
    """upload function is the main function page of this application. It takes uploaded
    image and run predict_image, then return the result back to front-end
    input -- None

    Parameters:
        img: uploaded image from user
        data: predicted result

    Returns:
         render_template("upload.html") -- render the page with variable img_path and result pass to front-end
    """
    if request.method == 'POST':
        img = request.files['img'].filename
        img = secure_filename(img)
        new_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '_' + img
        filename = photos.save(request.files['img'], name=new_name)
        data = predict_img(photos.path(filename))

        data['img_png'].save('static/'+filename[:-4]+'img_png.png')
        data['attention_map_png'].save('static/'+filename[:-4]+'attention_map_png.png')
        data['overlay_png'].save('static/'+filename[:-4]+'overlay_png.png')
        img_path_1 = url_for('static', filename=filename[:-4]+'img_png.png')
        img_path_2 = url_for('static', filename=filename[:-4]+'attention_map_png.png')
        img_path_3 = url_for('static', filename=filename[:-4]+'overlay_png.png')
        return flask.jsonify({"result": data['predictions'], "img_path_1": img_path_1,
                              "img_path_2": img_path_2, "img_path_3": img_path_3})
    else:
        img_path_1 = None
        img_path_2 = None
        img_path_3 = None
        result = []
    return render_template('upload.html', img_path_1=img_path_1,
                           img_path_2=img_path_2, img_path_3=img_path_3, result=result)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Initialize the data dictionary that will be returned from the view
    ensure the image is properly uploaded to the folder
    read in the image from string and preprocess the image
    call function: predict_img()

    Returns:
        flask.jsonify(data) -- json version of prediction values
    """
    data = {'state': False}
    if request.method == 'POST':
        img = request.files['image'].read()
        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, flags=1)
        data = predict_img(img)
    return flask.jsonify(data)


def predict_img(img):
    """run pytorch model prediction and get the output probablity and label result
    to be called by predict()

    Args:
        is_numpy -- pass to model.predict

    Returns:
         data -- prediction values
    """
    data = dict()
    start = time.time()
    result = model.predict(img)
    cost_time = time.time() - start
    data['predictions'] = list()

    m_predict = {'label': result[0], 'probability': ("%.2f" % result[1])}
    data['predictions'].append(m_predict)
    data['img_png'] = result[2]
    data['attention_map_png'] = result[3]
    data['overlay_png'] = result[4]
    data['state'] = True
    data['time'] = cost_time
    return data


@app.route('/')
def index():
    """render initial page

    Returns:
        render initial index.html loading page
    """
    return render_template('index.html')


@app.route('/front_page')
def front_page():
    """render initial page

    Returns:
        render front_page.html loading page
    """
    return render_template('front_page.html')


@app.route('/team_member')
def team_member():
    """render initial page

    Returns:
        render team_member.html loading page
    """
    return render_template('team_member.html')


def shutdown_server():
    """shutdown the server function function for Apache deployment
        """
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """run shuntdown_server() function

    Returns:
         'Server shutting down' -- message of server status
    """
    shutdown_server()
    return 'Server shutting down...'



