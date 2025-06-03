import argparse
import os
import datetime

import jiwer

# import wandb
from ultralytics import YOLO

from src.evaluate.evaluate_helper import test_results, test_results_v2

# from wandb.integration.ultralytics import add_wandb_callback

parser = argparse.ArgumentParser()

# example: C:/Users/simcor/dev/projects/master_student/nazrul/datasets/iam1-crossed_debug/data.yaml
# parser.add_argument("dataset_yaml")
parser.add_argument("dir_db")
parser.add_argument("path_model")
#parser.add_argument("log_dir")
# parser.add_argument("wandb_dir") # diff

# parser.add_argument('--batch_size', default=4, type=int)
# parser.add_argument('--nb_epochs_max', default=1, type=int)
# parser.add_argument('--debug_pc', default=0, type=int)
parser.add_argument('--img_height_yolo', default=640, type=int)
parser.add_argument('--img_weight_yolo', default=640, type=int)  # typo
# parser.add_argument('--patience_early_stopping', default=100, type=int)

# parser.add_argument('--name_exp', default="train_iam1-scratch", type=str)

#parser.add_argument('--iou_nms', default=0.5, type=float)
#parser.add_argument('--conf_filter', default=0.25, type=float)
parser.add_argument('--agnostic_nms', default=0, help="1 -> True", type=int)

args = parser.parse_args()
print(args)

print("Begin")
path_model = os.path.join(args.path_model)

model = YOLO(path_model)

agnostic_nms = False

if args.agnostic_nms == 1:
    agnostic_nms = True

for iou_v in [x / 100.0 for x in range(25, 50, 5)]:  # range(0.35, 0.65, 0.05):
    for conf_v in [x / 100.0 for x in range(10, 30, 5)]:  # range(0.2, 0.4, 0.05):
        print("iou: " + str(iou_v))
        print("conf_v: " + str(conf_v))

        predicted, labels, counter_sub_segmentation, counter_over_segmentation = test_results_v2(args.dir_db,
                                                                                                 model,
                                                                                                 args.img_height_yolo,
                                                                                                 args.img_weight_yolo,
                                                                                                 iou_v,
                                                                                                 conf_v,
                                                                                                 agnostic_nms)

        cer = jiwer.cer(labels, predicted)
        wer = jiwer.wer(labels, predicted)

        print(f"CER: {cer:.4f}, WER: {wer:.4f}")
        print(f"counter_sub_segmentation: {counter_sub_segmentation}, counter_over_segmentation: {counter_over_segmentation}")
