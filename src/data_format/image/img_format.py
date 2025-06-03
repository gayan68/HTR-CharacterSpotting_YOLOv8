import glob
import os
import shutil
import math

import numpy as np

from skimage import io
from skimage.transform import resize


def resized_one_set(dir_from, dir_to_save, height_r, width_r, pad_img_value, ext_img="png"):
    # gt format
    # <class> <x_center> <y_center> <width> <height>
    # <class>: The class label of the object.
    # <x_center>: The normalized x-coordinate of the bounding box center.
    # <y_center>: The normalized y-coordinate of the bounding box center.
    # <width>: The normalized width of the bounding box.
    # <height>: The normalized height of the bounding box.

    os.makedirs(dir_to_save, exist_ok=True)

    dir_from_img = os.path.join(dir_from, "images")
    dir_from_label = os.path.join(dir_from, "labels")

    dir_to_img = os.path.join(dir_to_save, "images")
    dir_to_label = os.path.join(dir_to_save, "labels")

    os.makedirs(dir_to_img, exist_ok=True)
    os.makedirs(dir_to_label, exist_ok=True)

    files_img = glob.glob(dir_from_img + '/*.' + ext_img, recursive=True)

    for one_file in files_img:
        # Get id file
        split_name = os.path.split(one_file)
        split_name = split_name[1].split(sep=".")  # Filename and extension
        id_file = split_name[0]

        img = io.imread(one_file, as_gray=True)

        # Get img size
        h_origin, w_origin = img.shape

        # Get rescale factor
        scale_y = float(height_r) / float(h_origin)
        scale_x = float(width_r) / float(w_origin)

        scale = min(scale_x, scale_y)

        # Downscale
        if scale < 1.0:
            width_new = math.floor(w_origin * scale)
            height_new = math.floor(h_origin * scale)

            img = resize(image=img, output_shape=(height_new, width_new)).astype(np.float32)
            # Reset in value 0 -> 25
            img *= 255.0
        else:
            scale = 1.0

        # Get img size after rezised
        h_resized, w_resized = img.shape

        h_pad = height_r - h_resized
        w_pad = width_r - w_resized

        # pad
        img_resized = np.pad(img, ((0, h_pad), (0, w_pad)), 'constant', constant_values=pad_img_value)

        path_label = os.path.join(dir_from_label, id_file + ".txt")

        if not os.path.exists(path_label):
            print("label does not exists: " + path_label)
            continue

        with open(path_label, "r", encoding="utf-8") as file:
            gt_origin_str = file.readlines()
            gt_resized = []

            for one_gt in gt_origin_str:
                # <class> <x_center> <y_center> <width> <height>
                # 5 0.09523809523809523 0.5 0.2 1.0
                one_gt_split = one_gt.split(" ")

                if len(one_gt_split) == 5:
                    # compute unormalize x_center, y_center>, width, height
                    # x_center_un = float(one_gt_split[1]) * w_origin
                    # y_center_un = float(one_gt_split[2]) * h_origin
                    # width_un = float(one_gt_split[3]) * w_origin
                    # height_un = float(one_gt_split[4]) * h_origin
                    #
                    # Rescale with aspect ratio -> same value normalized
                    # # Compute new normalized value
                    # # Change only if scale was applied
                    # # Introduce small shift to do precision operation
                    # x_center_save = (x_center_un * scale) / w_resized
                    # y_center_save = (y_center_un * scale) / h_resized
                    # width_save = (width_un * scale) / w_resized
                    # height_save = (height_un * scale) / h_resized
                    # #height_save = height_un / h_resized

                    gt_resized.append([one_gt_split[0], float(one_gt_split[1]), float(one_gt_split[2]),
                                       float(one_gt_split[3]), float(one_gt_split[4])])

            # Update bb gt
            if len(gt_resized) == 0:
                print("len(gt_resized) == 0")

            # Save new img
            path_img_new = os.path.join(dir_to_img, id_file + "." + ext_img)
            # scale can change type
            img_resized = img_resized.astype(np.uint8)
            io.imsave(path_img_new, img_resized)

            # Save new gt
            path_label_save = os.path.join(dir_to_label, id_file + ".txt")

            with open(path_label_save, "w", encoding="utf-8") as file:
                for one_gt in gt_resized:
                    file.write(one_gt[0])
                    file.write(" ")
                    file.write(str(one_gt[1]))
                    file.write(" ")
                    file.write(str(one_gt[2]))
                    file.write(" ")
                    file.write(str(one_gt[3]))
                    file.write(" ")
                    file.write(str(one_gt[4]))
                    file.write("\n")


def resized_all_db(dir_from, dir_to_save, height_r, width_r, pad_img_value, ext_img="png"):
    os.makedirs(dir_to_save, exist_ok=True)

    # Do it manually to update dir path
    # yaml_file = "data.yaml"
    #
    # path_from_yaml = os.path.join(dir_from, yaml_file)
    #
    # if os.path.exists(path_from_yaml):
    #     path_to_yaml = os.path.join(dir_to_save, yaml_file)
    #     shutil.copyfile(path_from_yaml, path_to_yaml)

    dir_from_save_train = os.path.join(dir_from, "train")
    dir_to_save_train = os.path.join(dir_to_save, "train")

    resized_one_set(dir_from_save_train, dir_to_save_train, height_r, width_r, pad_img_value, ext_img)

    dir_from_save_val = os.path.join(dir_from, "validation")
    dir_to_save_val = os.path.join(dir_to_save, "validation")

    resized_one_set(dir_from_save_val, dir_to_save_val, height_r, width_r, pad_img_value, ext_img)

    dir_from_save_test = os.path.join(dir_from, "test")
    dir_to_save_test = os.path.join(dir_to_save, "test")

    resized_one_set(dir_from_save_test, dir_to_save_test, height_r, width_r, pad_img_value, ext_img)


def correct_label_one_split(dir_from):
    dir_label_from = os.path.join(dir_from, "labels")

    dir_label_fix = os.path.join(dir_from, "labels_fix")

    os.makedirs(dir_label_fix, exist_ok=True)

    files_labels = glob.glob(dir_label_from + '/*.txt', recursive=True)

    for one_file in files_labels:
        # Get id file
        split_name = os.path.split(one_file)
        split_name = split_name[1].split(sep=".")  # Filename and extension
        id_file = split_name[0]

        with open(one_file, "r", encoding="utf-8") as file:
            gt_origin_str = file.readlines()
            gt_resized = []

            for one_gt in gt_origin_str:
                # <class> <x_center> <y_center> <width> <height>
                # 5 0.09523809523809523 0.5 0.2 1.0
                one_gt_split = one_gt.split(" ")

                if len(one_gt_split) == 5:

                    width = float(one_gt_split[3])

                    if width > 1.0:
                        print("fix width > 1")
                        width = 1.0

                    height = float(one_gt_split[4])

                    if height > 1.0:
                        print("fix height > 1")
                        height = 1.0

                    gt_resized.append([one_gt_split[0], float(one_gt_split[1]), float(one_gt_split[2]), width, height])

            # Update bb gt
            if len(gt_resized) == 0:
                print("len(gt_resized) == 0")

            # Save new gt
            path_label_save = os.path.join(dir_label_fix, id_file + ".txt")

            with open(path_label_save, "w", encoding="utf-8") as file:
                for one_gt in gt_resized:
                    file.write(one_gt[0])
                    file.write(" ")
                    file.write(str(one_gt[1]))
                    file.write(" ")
                    file.write(str(one_gt[2]))
                    file.write(" ")
                    file.write(str(one_gt[3]))
                    file.write(" ")
                    file.write(str(one_gt[4]))
                    file.write("\n")


def correct_labels_all_db(dir_from):
    dir_from_save_train = os.path.join(dir_from, "train")
    # dir_to_save_train = os.path.join(dir_to_save, "train")

    correct_label_one_split(dir_from_save_train)

    dir_from_save_val = os.path.join(dir_from, "validation")
    # dir_to_save_val = os.path.join(dir_to_save, "validation")

    correct_label_one_split(dir_from_save_val)

    dir_from_save_test = os.path.join(dir_from, "test")
    # dir_to_save_test = os.path.join(dir_to_save, "test")

    correct_label_one_split(dir_from_save_test)



if __name__ == "__main__":
    height_r = 128
    width_r = 224

    pad_img_value = 255


    # path_db_from = "C:/Users/simcor/dev/projects/master_student/nazrul/datasets/iam1-crossed_debug/"  #train/images/"
    # path_db_resized = "C:/Users/simcor/dev/projects/master_student/nazrul/datasets/iam1-crossed_debug_resized/"  #train/images/"
    #
    # resized_all_db(path_db_from, path_db_resized, height_r, width_r, pad_img_value)

    path_db_from = "C:/Users/simcor/dev/projects/master_student/nazrul/datasets/iam1/"  #train/images/"
    path_db_resized = "C:/Users/simcor/dev/projects/master_student/nazrul/datasets/iam1_resized_128_224/"  #train/images/"

    # resized_all_db(path_db_from, path_db_resized, height_r, width_r, pad_img_value)

    correct_labels_all_db(path_db_from)
