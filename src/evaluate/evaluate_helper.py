import os
from pathlib import Path

from tqdm import tqdm


def predict(img, model, img_height_yolo, img_weight_yolo, iou, conf, agnostic_nms):
    results = model(img, verbose=False, iou=iou, imgsz=(img_height_yolo, img_weight_yolo), conf=conf, agnostic_nms=agnostic_nms)
    # results = model(img, verbose=False, device="cpu")

    for result in results:

        boxes = result.boxes  # Boxes object for bounding box outputs
        letters = {box.xyxy[0][0].item(): result.names[box.cls.item()] for box in boxes}
        keys = list(letters.keys())
        keys.sort()
        letters_sorted = {i: letters[i] for i in keys}
        predicted_word = "".join(letters_sorted.values())

        # print(predicted_word)

        label_file = img.replace("png", "txt").replace("images", "labels")
        # print(label_file)

        f = open(label_file, "r")
        lines = f.readlines()
        # print(lines[0].split()[0])
        # true_word = {line.split()[0] for line in lines}
        true_word = ""
        for line in lines:
            true_word += result.names[int(line.split()[0])]

        # labels.append(true_word)
        # print(true_word)

        return predicted_word, true_word


def predict_v2(img_path, label_path, model, img_height_yolo, img_weight_yolo, iou, conf, agnostic_nms):
    results = model(img_path, verbose=False, iou=iou, imgsz=(img_height_yolo, img_weight_yolo), conf=conf, agnostic_nms=agnostic_nms)
    # results = model(img, verbose=False, device="cpu")

    for result in results:

        boxes = result.boxes  # Boxes object for bounding box outputs
        letters = {box.xyxy[0][0].item(): result.names[box.cls.item()] for box in boxes}
        keys = list(letters.keys())
        keys.sort()
        letters_sorted = {i: letters[i] for i in keys}
        predicted_word = "".join(letters_sorted.values())

        # print(predicted_word)

        # label_file = img.replace("png", "txt").replace("images", "labels")
        # print(label_file)

        f = open(label_path, "r")
        lines = f.readlines()
        # print(lines[0].split()[0])
        # true_word = {line.split()[0] for line in lines}
        true_word = ""
        for line in lines:
            true_word += result.names[int(line.split()[0])]

        # labels.append(true_word)
        # print(true_word)

        return predicted_word, true_word

def test_results(dataset_path,  model, img_height_yolo, img_weight_yolo, iou, conf, agnostic_nms):
    images_path = os.path.join(dataset_path, "test", "images")
    # labels_path = os.path.join(dataset_path, "test", "labels")

    predicted = []
    labels = []

    counter_sub_segmentation = 0
    counter_over_segmentation = 0

    images = os.listdir(images_path)

    for img in tqdm(images):
        if img.endswith('.png'):
            img = os.path.join(images_path, img)

            predicted_word, true_word = predict(img, model, img_height_yolo, img_weight_yolo, iou, conf, agnostic_nms)

            if len(predicted_word) < len(true_word):
                counter_sub_segmentation += 1
            if len(predicted_word) > len(true_word):
                counter_over_segmentation += 1

            predicted.append(predicted_word)
            labels.append(true_word)

    return predicted, labels, counter_sub_segmentation, counter_over_segmentation


def test_results_v2(images_path, dir_label,  model, img_height_yolo, img_weight_yolo, iou, conf, agnostic_nms):
    #images_path = os.path.join(dataset_path, "images")
    # # labels_path = os.path.join(dataset_path, "test", "labels")

    predicted = []
    labels = []

    counter_sub_segmentation = 0
    counter_over_segmentation = 0

    images = os.listdir(images_path)
    print("nb images:" + str(len(images)))

    for img in tqdm(images):
        if img.endswith('.png'):
            img = os.path.join(images_path, img)
            id_item = Path(img).stem.split('.')[0]
            label_path = os.path.join(dir_label, id_item + ".txt")

            predicted_word, true_word = predict_v2(img, label_path, model, img_height_yolo, img_weight_yolo, iou, conf, agnostic_nms)

            if len(predicted_word) < len(true_word):
                counter_sub_segmentation += 1
            if len(predicted_word) > len(true_word):
                counter_over_segmentation += 1

            predicted.append(predicted_word)
            labels.append(true_word)

    return predicted, labels, counter_sub_segmentation, counter_over_segmentation
