from flask import Flask,jsonify,request, render_template
import io
from PIL import Image
from torch_utils import get_prediction, transform_image

app = Flask(__name__)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def render_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            image = io.BytesIO(img_bytes)
            image = Image.open(image)
            tensor = transform_image(image)
            prediction = get_prediction(tensor)
            data = {'prediction': prediction}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})