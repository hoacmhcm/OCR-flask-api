import torch

def convert_yolo_to_list(yolo_tensor):
    # Assuming yolo_tensor is a PyTorch tensor, convert it to a list of bounding boxes
    yolo_boxes = yolo_tensor.tolist()
    return yolo_boxes

def convert_list_to_yolo(bounding_boxes):
    # Assuming bounding_boxes is a list of bounding boxes, convert it to a tensor
    # yolo_tensor = torch.tensor(bounding_boxes, device='cuda:0')
    yolo_tensor = torch.tensor(bounding_boxes, device='cpu')
    return yolo_tensor

def sorting_yolo_bounding_boxes(yolo_boxes):
    final_sorted_list = []

    # print(yolo_boxes)

    while True:
        try:
            new_sorted_text = []

            # Sort YOLO bounding boxes by their y-coordinate (top to bottom)
            sorted_boxes = sorted(yolo_boxes, key=lambda box: box[1])

            # Initialize variables for the current line
            current_line = [sorted_boxes[0]]
            current_line_y = (sorted_boxes[0][1] + sorted_boxes[0][3]) / 2

            # Iterate through the remaining boxes
            for box in sorted_boxes[1:]:
                box_y = (box[1] + box[3]) / 2

                # If the box is within a threshold distance from the current line, add it to the line
                if abs(box_y - current_line_y) <= 20:
                    current_line.append(box)
                else:
                    # Sort the boxes in the current line from left to right
                    current_line = sorted(current_line, key=lambda box: box[0])
                    new_sorted_text.extend(current_line)
                    current_line = [box]
                    current_line_y = box_y

            # Sort the last line (if any) from left to right
            if current_line:
                current_line = sorted(current_line, key=lambda box: box[0])
                new_sorted_text.extend(current_line)

            final_sorted_list.append(new_sorted_text)

            # Remove the sorted boxes from the list
            yolo_boxes = [box for box in yolo_boxes if box not in new_sorted_text]

        except Exception as e:
            print(e)
            break

    return final_sorted_list
