import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def str_pixel_width_calculator(string_in):
    char_widths = [0] * len(string_in)

    for ii in range(len(string_in)):
        letter_width = 0

        char = string_in[ii]
        #print(char)
        if char in ['.', ',', "'"]:
            letter_width = 8
        elif char in ["0","1","2","3","4","5","6","7","8","9"]:
            letter_width = 15
        elif char in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']:
            letter_width = 23
        elif char in ['I', 'J', '-']:
            letter_width = 15 
        elif char == 'M':
            letter_width = 26 
        elif char == 'W':
            letter_width = 39
        elif char in ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'k', 'n', 'o', 'p', 'q', 's', 'u', 'v', 'x', 'y', 'z']:
            letter_width = 15  
        elif char in ['i', 'l']:
            letter_width = 8 
        elif char in ['r', 't', 'j', 'f']:
            letter_width = 12 
        elif char in ['m', 'w']:
            letter_width = 24 
        elif char in [';', '!', '?', '(', ')', '"', ":", "&", "#", "*", " ", '/', "+"]:
            letter_width = 15

        char_widths[ii] = letter_width

    return char_widths

def add_width_buffer(char_widths):
    buffer_char_widths = char_widths[:]
    for width in range(len(char_widths)):
        if width == 0 and width == len(char_widths):
            buffer_char_widths[width] = 1 #To fix single letters
        elif width == 0 or width == len(char_widths):
            buffer_char_widths[width] = char_widths[width] * 1.05 
        else:
            buffer_char_widths[width] = char_widths[width] * 1.1
    return buffer_char_widths

def add_width_buffer2(char_widths, Wh, Wp):
    buffer_char_widths = []
    for i, Xp in enumerate(char_widths):
        end = True if (i == 0) or (i == len(char_widths)-1) else False
        Xe = estimated_width_for_ht(Xp,Wh,Wp,end)
        buffer_char_widths.append(Xe)
    return buffer_char_widths

def estimated_width_for_ht(Xp,Wh,Wp,end):
    n = 0.2
    dev = 2 if end else 1
    Xe = int(Xp * ((Wh / Wp) + (n/dev)))
    return Xe



###################### MAIN ##############################################

root = "../DATASETS/IAM/YOLO"
split = "test"
image_root = f"{root}/{split}/clean_hc/images"
save_dir = f"yolo_labels2/{split}"

yolo_classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '?']

with open(os.path.join(root, split, 'clean_hc', 'gt.txt'), 'r') as f:
    for line in f:
        img_id, transcr = line.strip().split(' ')[0], ' '.join(line.strip().split(' ')[1:])
        picture_path = f"{root}/{split}/clean_hc/images/{img_id}.png"

        image = cv2.imread(picture_path)
        im_h, im_w, _ = image.shape

        Wh = int((im_w / im_h) * 25)  #In above, we decided to have 25 as the printed text height based on the above printed text image
        #print(f"Original width: {im_w} | New image width: {Wh}")

        char_widths = str_pixel_width_calculator(transcr)
        Wp = np.sum(char_widths)
        char_widths_estimates = add_width_buffer2(char_widths, Wh, Wp)

        #for the normalized characters
        char_widths_normalized = [width / sum(char_widths) for width in char_widths]
        char_widths_estimates_normalized = [width / Wh for width in char_widths_estimates]

        char_positions = [0] + [sum(char_widths_normalized[:j]) for j in range(1, len(char_widths_normalized))]

        ##############################################################################
        yolo_annotation_path = f"{save_dir}/{img_id}.txt"

        with open(yolo_annotation_path, "a") as yolo_annotation_file:
            for j, char in enumerate(transcr):
                if (j == 0):
                    x_center = char_positions[j] + char_widths_estimates_normalized[j] / 2
                elif (j == len(transcr)-1):
                    x_center = char_positions[j] + char_widths_normalized[j] - char_widths_estimates_normalized[j] / 2
                else:
                    x_center = char_positions[j] + (char_widths_normalized[j] / 2)
                    
                y_center = 0.5  # Assuming center alignment vertically
                box_width = char_widths_estimates_normalized[j]
                box_height = 1.0  # Assuming the character spans the full height of the image
                class_label = yolo_classes.index(char)

                yolo_annotation_file.write(f"{class_label} {x_center} {y_center} {box_width} {box_height}\n")

        print(f"{transcr}")
