import os
import cv2

def write_image_to_output(index, image):
    # Define the folder name and the output file name
    folder_name = "output"
    file_name = f'image_{index}.jpg'

    # Create the output folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Construct the full file path including the folder
    output_path = os.path.join(folder_name, file_name)

    # Save the image to the specified folder and filename
    cv2.imwrite(output_path, image)
