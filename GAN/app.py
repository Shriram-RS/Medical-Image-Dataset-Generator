import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from GAN import train_gan
import zipfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GAN_FOLDER'] = 'static/gan_images'
app.config['ZIP_FOLDER'] = 'static/zips'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['GAN_FOLDER']):
    os.makedirs(app.config['GAN_FOLDER'])

if not os.path.exists(app.config['ZIP_FOLDER']):
    os.makedirs(app.config['ZIP_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        epochs = int(request.form.get('epochs', 4000))
        train_gan(file_path, epochs)
        zip_path = os.path.join(app.config['ZIP_FOLDER'], 'gan_images.zip')
        create_zip(app.config['GAN_FOLDER'], zip_path)
        return render_template('index.html', message='GAN images have been generated and saved in static/gan_images. The ZIP file is in static/zips.')

def create_zip(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)

if __name__ == '__main__':
    app.run(debug=True)
