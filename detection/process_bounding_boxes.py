import cv2
import numpy as np
from detection.sort_bounding_boxes import convert_yolo_to_list, sorting_yolo_bounding_boxes, convert_list_to_yolo
from detection.save_image import write_image_to_output


def process_bounding_boxes(results, original_image_url, max_boxes_per_image=10, spacing=10, output_dir=None):
    bounding_boxes = []

    # Load the original image (assuming it's the same for all boxes)
    if len(results) > 0:
        original_image = cv2.imread(original_image_url)
    else:
        raise ValueError("No bounding boxes to process.")

    # Process results list
    for result in results:
        bounding_boxes = result.boxes.xyxy

    # Convert the YOLO format tensor to a list of bounding boxes
    yolo_boxes_list = convert_yolo_to_list(bounding_boxes)

    # Sort the YOLO format bounding boxes
    sorted_yolo_boxes = sorting_yolo_bounding_boxes(yolo_boxes_list)

    # Convert the sorted bounding boxes back to a YOLO format tensor
    sorted_bboxes = convert_list_to_yolo(sorted_yolo_boxes[0])

    # Initialize variables
    current_box_index = 0
    current_image = None
    current_x = 0
    current_image_index = 0
    # index = 0

    # Iterate through sorted bounding boxes
    for x1, y1, x2, y2 in sorted_bboxes:
        # Ensure the coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Crop the text region from the original image
        text_region = original_image[y1:y2, x1:x2]

        # Calculate the width and height of the bounding box
        width, height = x2 - x1, y2 - y1

        # Resize the text region to match the width and height of the bounding box
        text_region = cv2.resize(text_region, (width, height))

        # If a new image needs to be created or the maximum boxes per image is reached
        if current_image is None or current_box_index == max_boxes_per_image:
            current_box_index = 0
            current_x = 10

            # Create a new image with white background
            new_image_height = height + 2 * spacing
            new_image_width = (max_boxes_per_image * (width + spacing)) + spacing
            current_image = np.ones((new_image_height, new_image_width, 3), dtype=np.uint8) * 255

            # index += 1

        # Calculate the Y-coordinate for centering vertically
        current_y = spacing

        # Calculate the maximum width and height for the target area in the current image
        max_width = new_image_width - current_x
        max_height = new_image_height - current_y

        # Check if available space is greater than zero
        if max_width > 0 and max_height > 0:
            # If the text region is wider or taller than the available space, resize it accordingly
            if width > max_width or height > max_height:
                # Calculate the aspect ratio to maintain the text region's proportions
                aspect_ratio = float(width) / float(height)

                if width > max_width:
                    text_region = cv2.resize(text_region, (max_width, int(max_width / aspect_ratio)))
                elif height > max_height:
                    text_region = cv2.resize(text_region, (int(max_height * aspect_ratio), max_height))

            # Paste the text region onto the current image with spacing
            current_image[current_y:current_y + text_region.shape[0],
            current_x:current_x + text_region.shape[1]] = text_region

        # Update coordinates and current image index
        current_x += text_region.shape[1] + spacing
        current_box_index += 1

        # If the maximum boxes per image is reached, save the current image
        if current_box_index == max_boxes_per_image:
            if output_dir:
                write_image_to_output(current_image_index, current_image, output_dir)
                current_image_index += 1

    # Save the last image if needed (in case it doesn't reach the maximum)
    if current_box_index > 0 and current_box_index != max_boxes_per_image and output_dir:
        write_image_to_output(current_image_index, current_image, output_dir)


# results = run_yolo_inference('model/model.pt', 'img.png')
# # # Example usage:
# process_bounding_boxes(results, 'img.png', max_boxes_per_image=10, spacing=10, output_dir='/path/to/output')
