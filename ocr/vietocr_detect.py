from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import os


def perform_ocr(image_path):
    # Load the configuration
    config = Cfg.load_config_from_name('vgg_transformer')

    # Modify configuration settings
    config['cnn']['pretrained'] = False
    # config['device'] = 'cuda:0'
    config['device'] = 'cpu'

    # Initialize the OCR detector
    detector = Predictor(config)

    # Open the image
    img = Image.open(image_path)

    # # Display the image using matplotlib
    # plt.imshow(img)
    # plt.show()

    # Perform OCR on the image
    result = detector.predict(img)

    return result

def list_and_sort_image_files(folder_path):
    # List the image files in the folder and sort them by name (ascending order)
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")], key=lambda x: int(x.split('_')[1].split('.')[0]))
    return image_files

def perform_ocr_and_combine_text_for_sorted_images(folder_path):
    image_files = list_and_sort_image_files(folder_path)
    recognized_text = []

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        print(image_path)
        text = perform_ocr(image_path)  # Assuming perform_ocr is defined elsewhere
        if text:
            recognized_text.append(text)
        # Delete the image file after OCR
        os.remove(image_path)

    paragraph = " ".join(recognized_text)
    return paragraph

# Usage:
# folder_path = '/content/output/'
# combined_text = perform_ocr_and_combine_text_for_sorted_images(folder_path)
# print(combined_text)

