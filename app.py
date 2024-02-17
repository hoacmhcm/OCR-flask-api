import os
import logging
from flask import Flask, jsonify, request, render_template, redirect
from flask_cors import CORS
from detection.detect import run_yolo_inference, load_yolo_model
from detection.process_bounding_boxes import process_bounding_boxes
from ocr.vietocr_detect import perform_ocr_and_combine_text_for_sorted_images, load_vietocr_model
from utils.utils_function import remove_images_from_folder
import time
import requests

app = Flask(__name__)

model_folder = os.path.join('detection', 'model')
# Define the path to the 'model.pt' file within the 'model' folder
model_file_path = os.path.join(model_folder, 'model.pt')
yolo_model = load_yolo_model(model_file_path)

detector = load_vietocr_model()

# Initialize CORS extension with specific options
cors = CORS(app, resources={
    r"/api/*": {
        "origins": ["https://localhost:8080", "https://jitsi-hoacm2.dedyn.io"]
    }
})

# Configure logging to write to a file with timestamps and log levels
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler('log_module.txt')
file_handler.setFormatter(log_formatter)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# api_url = 'http://localhost:5000/api/store-info'
api_url = 'https://store-ocr-info-flask-api.onrender.com/api/store-info'

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
YOLO_OUTPUT_FOLDER = os.path.join('runs', 'detect', 'predict')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Log all requests
@app.after_request
def after_request(response):
    log_message = f"{request.method} {request.url} {response.status_code}"
    app.logger.info(log_message)
    return response


# Log errors
@app.errorhandler(Exception)
def log_error(error):
    log_message = f"Internal Server Error: {str(error)}"
    app.logger.error(log_message)
    return jsonify({"error": "Internal Server Error"}), 500


@app.route("/api/test", methods=["GET"])
def test():
    # List all files in the image directory
    image_files = os.listdir(UPLOAD_FOLDER)
    image_path = os.path.join(UPLOAD_FOLDER, image_files[0])

    output_dir = os.path.join('staticFiles', 'process_bounding_boxes')

    results = run_yolo_inference(yolo_model, image_path, save=True, show=True)

    print(results)

    process_bounding_boxes(results, image_path, max_boxes_per_image=10, spacing=10,
                           output_dir=output_dir)

    combined_text = perform_ocr_and_combine_text_for_sorted_images(output_dir, detector)
    print(combined_text)
    # Delete the image file after OCR
    # os.remove(image_path)
    # remove_images_from_folder(output_dir)

    return combined_text


@app.route("/api/upload-image", methods=["POST"])
def upload_image():
    output_dir = os.path.join('staticFiles', 'process_bounding_boxes')

    # Check if the post request has the file part
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Extract the string data from the form
    name = request.form.get('name', '')
    roomName = request.form.get('roomName', '')
    sessionId = request.form.get('sesssionId', '')
    # if not name:
    #     return 'No name provided', 400

    uid = request.form.get('uid', '')
    if not uid:
        return 'No name provided', 400

    if request.files:
        image = request.files["file"]
        image.save(os.path.join(app.config["UPLOAD_FOLDER"], image.filename))

        # # List all files in the image directory
        image_files = os.listdir(UPLOAD_FOLDER)
        image_path = os.path.join(UPLOAD_FOLDER, image_files[0])

        # Start the timer
        start_time = time.time()

        results = run_yolo_inference(yolo_model, image_path, save=True)

        yolo_output_path = None
        # Process results list
        for result in results:
            yolo_output_path = result.save_dir

        print(yolo_output_path)

        process_bounding_boxes(results, image_path, max_boxes_per_image=10, spacing=10,
                               output_dir=output_dir)

        combined_text = perform_ocr_and_combine_text_for_sorted_images(output_dir, detector)
        # Stop the timer
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Prepare the data
        data = {
            'name': name,
            'room_name': roomName,
            'session_id': sessionId,
            'ocr_time': elapsed_time,
            'ocr_result': combined_text,
        }

        print(yolo_output_path)

        yolo_output_file = os.listdir(yolo_output_path)
        yolo_output_file_path = os.path.join(yolo_output_path, yolo_output_file[0])

        # Prepare the files
        files = {
            'originImage': ('origin_image.png', open(image_path, 'rb')),
            'yoloImage': ('yolo_image.png', open(yolo_output_file_path, 'rb')),
            # 'processImage': ('process_image.jpg', open(process_image_path, 'rb')),
        }

        response = requests.post(api_url, data=data, files=files)

        print(f"OCR time: {elapsed_time} seconds")
        remove_images_from_folder(UPLOAD_FOLDER)
        remove_images_from_folder(output_dir)
        print(combined_text)
        return combined_text


if __name__ == '__main__':
    # app.run(port=4000, debug=True)
    app.run()
