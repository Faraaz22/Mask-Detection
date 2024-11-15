import os
import numpy as np
from flask import Flask, render_template_string, request, redirect, send_from_directory, url_for, flash
from tensorflow.keras.models import load_model
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for using session or flash messages

# Load the trained model
model = load_model('mask_model.h5')

# Prepare the uploaded image for prediction
def prepare_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_rescaled = img_resized / 255.0
    return np.reshape(img_rescaled, [1, 128, 128, 3])

# Set upload folder and allowed file types
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configure Flask to allow file uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Get result based on prediction
def getResult(pred_label):
    if pred_label == 0:
        return 'No Mask'
    else:
        return 'Mask'

@app.route('/', methods=['GET'])
def index():
    result = request.args.get('result')
    image_path = request.args.get('image_path')

    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mask Detection using CNN</title>
        <style>
            body { font-family: Arial, sans-serif; margin-left: 35%; }
            h1 { color: #333; }
            .upload-form { margin-top: 20px; }
            .upload-form input[type="file"] { padding: 10px; }
            .upload-form button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            .result { margin-top: 30px;  }
            .result img { max-width: 300px; margin-top: 10px;}
        </style>
    </head>
    <body>
        <h1>Mask Detection using CNN</h1>
        <p>Upload an image to check if the person is wearing a mask or not.</p>

        <div class="upload-form">
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" required>
                <button type="submit">Upload Image</button>
            </form>
        </div>

        {% if result %}
        <div class="result">
            <h2>{{ result }}</h2>
            <img src="{{ image_path }}" alt="Uploaded Image">
        </div>
        {% endif %}
    </body>
    </html>
    '''

    return render_template_string(html_content, result=result, image_path=image_path)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']

        # Create uploads directory 
        upload_folder = os.path.join(os.path.dirname(__file__), 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the uploaded file 
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        # Prepare the image 
        image = prepare_image(file_path)
        prediction = model.predict(image)
        pred_label = np.argmax(prediction)

        result = getResult(pred_label)
        image_path = url_for('uploaded_file', filename=secure_filename(f.filename))
        return redirect(url_for('index', result=result, image_path=image_path))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
