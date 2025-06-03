from ultralytics import YOLO


img_height_yolo = 640
img_weight_yolo = 640

conf_t = 0.25
iou_low = 0.5
iou_high = 0.5

log_dir = "C:/Users/simcor/dev/logs/MasterStudent/Nazrul/nms/"

#Best model CER: 0.14696384972828228, WER: 0.3775839793281654
path_model = "C:/Users/simcor/dev/logs/MasterStudent/Nazrul/2024-10-15_15_12_05_049936/weights/best.pt"
# path_db = "C:/Users/simcor/dev/projects/master_student/nazrul/datasets/iam1-crossed/"

id_item = "h07-060b-03-06"
path_img = "C:/Users/simcor/dev/projects/master_student/nazrul/datasets/iam1-crossed/validation/images/" + id_item + ".png"
path_label = "C:/Users/simcor/dev/projects/master_student/nazrul/datasets/iam1-crossed/validation/labels/" + id_item + ".txt"

model = YOLO(path_model)

# Before NMS -> iou 1
results = model(path_img, verbose=True, iou=1, imgsz=(img_height_yolo, img_weight_yolo), conf=conf_t)

for result in results:

    boxes = result.boxes  # Boxes object for bounding box outputs

    conf = boxes.conf
    x_letter_conf = {box.xyxy[0][0].item(): [result.names[box.cls.item()], box.conf.item()] for box in boxes}
    keys = list(x_letter_conf.keys())
    keys.sort()  # sort by x
    letters_sorted = {i: x_letter_conf[i][0] for i in keys}
    conf_sorted = {i: x_letter_conf[i][1] for i in keys}
    predicted_word = "".join(letters_sorted.values())

    for l, c in zip(letters_sorted.values(), conf_sorted.values()):
        print(l + " " + str(c))

    # Ground truth
    f = open(path_label, "r")
    lines = f.readlines()

    true_word = ""
    for line in lines:
        true_word += result.names[int(line.split()[0])]

    print("Gt  : " + true_word)
    print("Pred: " + predicted_word)

    # result.show()  # display to screen
    result_img = log_dir + "result_before_nms_conf_" + str(conf_t) + ".jpg"
    result.save(filename=result_img)  # save to disk

# NMS agnostic class   agnostic_nms defaut False
results = model(path_img, verbose=True, iou=iou_high, imgsz=(img_height_yolo, img_weight_yolo), conf=conf_t, agnostic_nms=True)

print("NMS agnostic:")
for result in results:

    boxes = result.boxes  # Boxes object for bounding box outputs

    conf = boxes.conf
    x_letter_conf = {box.xyxy[0][0].item(): [result.names[box.cls.item()], box.conf.item()] for box in boxes}
    keys = list(x_letter_conf.keys())
    keys.sort()  # sort by x
    letters_sorted = {i: x_letter_conf[i][0] for i in keys}
    conf_sorted = {i: x_letter_conf[i][1] for i in keys}
    predicted_word = "".join(letters_sorted.values())

    for l, c in zip(letters_sorted.values(), conf_sorted.values()):
        print(l + " " + str(c))

    # Ground truth
    f = open(path_label, "r")
    lines = f.readlines()

    true_word = ""
    for line in lines:
        true_word += result.names[int(line.split()[0])]

    print("Gt  : " + true_word)
    print("Pred: " + predicted_word)

    # result.show()  # display to screen
    result_img = log_dir + "result_nms_agnostic_iou_" + str(iou_high) + "_conf_" + str(conf_t) + ".jpg"
    result.save(filename=result_img)  # save to disk

# After NMS
print("NMS:")
results = model(path_img, verbose=True, iou=iou_high, imgsz=(img_height_yolo, img_weight_yolo), conf=conf_t, agnostic_nms=False)

for result in results:

    boxes = result.boxes  # Boxes object for bounding box outputs

    letters = {box.xyxy[0][0].item(): result.names[box.cls.item()] for box in boxes}
    keys = list(letters.keys())
    keys.sort()  # sort by x
    letters_sorted = {i: letters[i] for i in keys}
    predicted_word = "".join(letters_sorted.values())

    conf = boxes.conf
    x_letter_conf = {box.xyxy[0][0].item(): [result.names[box.cls.item()], box.conf.item()] for box in boxes}
    keys = list(x_letter_conf.keys())
    keys.sort()  # sort by x
    letters_sorted = {i: x_letter_conf[i][0] for i in keys}
    conf_sorted = {i: x_letter_conf[i][1] for i in keys}
    predicted_word = "".join(letters_sorted.values())

    for l, c in zip(letters_sorted.values(), conf_sorted.values()):
        print(l + " " + str(c))

    # Ground truth
    f = open(path_label, "r")
    lines = f.readlines()

    true_word = ""
    for line in lines:
        true_word += result.names[int(line.split()[0])]

    print("Gt  : " + true_word)
    print("Pred: " + predicted_word)

    # result.show()  # display to screen
    result_img = log_dir + "result_iou_" + str(iou_high) + "_conf_" + str(conf_t) + ".jpg"
    result.save(filename=result_img)  # save to disk
