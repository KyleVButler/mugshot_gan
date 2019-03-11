from gan_experiment.definitions import ROOT_DIR
import subprocess
from shutil import copy
import os

data_path = ROOT_DIR + '/data/'
if not os.path.exists(data_path):
    os.makedirs(data_path)


link = 'https://s3.amazonaws.com/nist-srd/SD18/sd18.zip'
subprocess.run(["curl", "-o", ROOT_DIR + '/data.zip', link])
subprocess.run(["unzip", ROOT_DIR + '/data.zip', "-d", ROOT_DIR + '/data'])

def copy_valid_images(save_path, copy_path):
    # list of paths to front profile images
    image_list = []
    for dirname, dirnames, filenames in os.walk(save_path):
        # print path to all filenames.
        for filename in filenames:
            full_file = os.path.join(dirname, filename)
            print(filename)
            if '_F.png' in full_file:
                image_list.append(full_file)

    if not os.path.exists(copy_path):
        os.makedirs(copy_path)
    for i, fn in enumerate(image_list):
        copy(fn, copy_path + str(i) + '.png')

    return image_list

copy_img_path = ROOT_DIR + '/image_folder/images/'
copy_valid_images(data_path, copy_img_path)

