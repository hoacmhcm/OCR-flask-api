from flask import Flask, jsonify, request, render_template, redirect
from flask_cors import CORS
import os

app = Flask(__name__)

# Initialize CORS extension with specific options
cors = CORS(app, resources={r"/api/*": {"origins": "https://localhost:8080"}})

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/api/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({'message': 'No file part'}), 400

        file = request.files['file']


        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400

        if request.files:
            image = request.files["file"]
            print(image)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], image.filename))
            return os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
    return render_template("upload_image.html")


if __name__ == '__main__':
    app.run(port=4000, debug=True)
