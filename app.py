from flask import Flask, jsonify, request, render_template, redirect
from flask_cors import CORS
import os
from detection.detect import run_yolo_inference
from detection.process_bounding_boxes import process_bounding_boxes
from ocr.vietocr_detect import perform_ocr_and_combine_text_for_sorted_images

app = Flask(__name__)

# Initialize CORS extension with specific options
cors = CORS(app, resources={r"/api/*": {"origins": "https://localhost:8080"}})

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/api/test", methods=["GET"])
def test():
    # List all files in the image directory
    image_files = os.listdir(UPLOAD_FOLDER)
    image_path = os.path.join(UPLOAD_FOLDER, image_files[2])

    model_folder = os.path.join('detection', 'model')
    # Define the path to the 'model.pt' file within the 'model' folder
    model_file_path = os.path.join(model_folder, 'model.pt')

    output_dir = os.path.join('staticFiles', 'process_bounding_boxes')

    results = run_yolo_inference(model_file_path, image_path)

    process_bounding_boxes(results, image_path, max_boxes_per_image=10, spacing=10,
                           output_dir=output_dir)

    combined_text = perform_ocr_and_combine_text_for_sorted_images(output_dir)
    print(combined_text)
    return render_template("upload_image.html")


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
    app.run()
