import os
import shutil


def remove_images_from_folder(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                # If you also want to remove directories, uncomment the following line
                shutil.rmtree(file_path)
                pass
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def list_and_sort_image_files(folder_path):
    # List the image files in the folder and sort them by name (ascending order)
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")],
                         key=lambda x: int(x.split('_')[1].split('.')[0]))

    return image_files
