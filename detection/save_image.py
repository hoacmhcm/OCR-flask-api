import os
import cv2

PROCESS_FOLDER = os.path.join('staticFiles', 'process_bounding_boxes')


def write_image_to_output(index, image, output_dir=PROCESS_FOLDER):
    # Create the output folder if it doesn't exist
    if not os.path.exists(PROCESS_FOLDER):
        os.makedirs(PROCESS_FOLDER)

    file_name = f'image_{index}.jpg'

    # Construct the full file path including the folder
    output_path = os.path.join(PROCESS_FOLDER, file_name)

    # Save the image to the specified folder and filename
    cv2.imwrite(output_path, image)
