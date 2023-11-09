import os
from flask import Flask, jsonify, request, render_template, redirect
from flask_cors import CORS
from detection.detect import run_yolo_inference, load_yolo_model
from detection.process_bounding_boxes import process_bounding_boxes
from ocr.vietocr_detect import perform_ocr_and_combine_text_for_sorted_images, load_vietocr_model
from utils.utils_function import remove_images_from_folder


app = Flask(__name__)

model_folder = os.path.join('detection', 'model')
# Define the path to the 'model.pt' file within the 'model' folder
model_file_path = os.path.join(model_folder, 'model.pt')
yolo_model = load_yolo_model(model_file_path)

detector = load_vietocr_model()

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
    image_path = os.path.join(UPLOAD_FOLDER, image_files[0])

    output_dir = os.path.join('staticFiles', 'process_bounding_boxes')

    results = run_yolo_inference(yolo_model, image_path)

    process_bounding_boxes(results, image_path, max_boxes_per_image=10, spacing=10,
                           output_dir=output_dir)

    combined_text = perform_ocr_and_combine_text_for_sorted_images(output_dir, detector)
    print(combined_text)
    # Delete the image file after OCR
    os.remove(image_path)
    remove_images_from_folder(output_dir)

    return combined_text


@app.route("/api/upload-image", methods=["POST"])
def upload_image():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Extract the string data from the form
    name = request.form.get('name', '')
    if not name:
        return 'No name provided', 400

    uid = request.form.get('uid', '')
    if not uid:
        return 'No name provided', 400

    if request.files:
        image = request.files["file"]
        image.save(os.path.join(app.config["UPLOAD_FOLDER"], image.filename))

        # # List all files in the image directory
        image_files = os.listdir(UPLOAD_FOLDER)
        image_path = os.path.join(UPLOAD_FOLDER, image_files[0])

        output_dir = os.path.join('staticFiles', 'process_bounding_boxes')

        results = run_yolo_inference(yolo_model, image_path, save=False)

        process_bounding_boxes(results, image_path, max_boxes_per_image=10, spacing=10,
                               output_dir=output_dir)

        combined_text = perform_ocr_and_combine_text_for_sorted_images(output_dir, detector)
        remove_images_from_folder(UPLOAD_FOLDER)
        remove_images_from_folder(output_dir)
        # print(combined_text)
        return combined_text


if __name__ == '__main__':
    # app.run(port=4000, debug=True)
    app.run()
