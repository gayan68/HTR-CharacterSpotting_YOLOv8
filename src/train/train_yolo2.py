import argparse
import os
import datetime

import jiwer
#from wandb import run

#import wandb
from ultralytics import YOLO

import sys
import os
import pandas as pd
# Add the src directory (the parent of the current script) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluate.evaluate_helper import test_results, test_results_v2

# from wandb.integration.ultralytics import add_wandb_callback

parser = argparse.ArgumentParser()

# example: C:/Users/simcor/dev/projects/master_student/nazrul/datasets/iam1-crossed_debug/data.yaml
parser.add_argument("dataset_yaml")
# parser.add_argument("dir_db")
parser.add_argument("dir_label_val")
parser.add_argument("dir_label_test_clean")
parser.add_argument("log_dir")
# parser.add_argument("wandb_dir") # diff

parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--nb_epochs_max', default=1, type=int)
parser.add_argument('--debug_pc', default=0, type=int)
parser.add_argument('--img_height_yolo', default=512, type=int)
parser.add_argument('--img_weight_yolo', default=512, type=int)  # typo
parser.add_argument('--patience_early_stopping', default=100, type=int)

parser.add_argument('--name_exp', default="train_iam1-scratch", type=str)

parser.add_argument('--name_model', default="yolov8x.yaml", type=str)

parser.add_argument('--agnostic_nms', default=1, help="1 -> True", type=int)

parser.add_argument('--verbose_yolo', default=0, type=int)

# Augmentation
# https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters
parser.add_argument('--degrees', default=0, type=float)
parser.add_argument('--shear', default=0, type=float)
parser.add_argument('--translate', default=0, type=float)

parser.add_argument('--gpu_device', default="cuda:0", type=str)
parser.add_argument('--trained_on', default="", type=str)
parser.add_argument('--model_id', default="", type=str)

args = parser.parse_args()
print(args)

#parser.add_argument("--path_model", default="", help="path of pretrained model", type=str)
#
# # https://docs.ultralytics.com/integrations/weights-biases/#how-does-weights-biases-help-in-optimizing-yolo11-models
# # cf. comments
#
# # Wandb Key API 926842974e2185e359a648fc1b2030c5c813045b
# # Setup Wandb for log
# # if not args.debug_pc:
# #     wandb.login(key="926842974e2185e359a648fc1b2030c5c813045b")
# #     wandb.init(project="character_spotting", entity="htr-analysis", dir=args.wandb_dir)  # , dir=args.log_dir
# #     wandb.config = {
# #         "epochs": args.nb_epochs_max,
# #         "batch_size": args.batch_size
# #     }
# #     print("run name wand : " + str(wandb.run.name))
# # else:
# #     print(os.environ)
# #     os.environ['WANDB_DISABLED'] = 'disable'  # ko true, to test disabled
#
# wandb.init(project="character_spotting", entity="htr-analysis", dir=args.log_dir)  # , dir=args.log_dir
# # wandb.config = {
# #     "epochs": args.nb_epochs_max,
# #     "batch_size": args.batch_size
# # }

# print("run name wand : " + str(wandb.run.name))

import yaml

data_yaml = None
with open(args.dataset_yaml) as stream:
    try:
        data_yaml = yaml.safe_load(stream)
        print(data_yaml)
    except yaml.YAMLError as exc:
        print(exc)

# https://docs.ultralytics.com/integrations/weights-biases/#usage-training-yolo11-with-weights-biases
#wandb.login(key="926842974e2185e359a648fc1b2030c5c813045b")

model = YOLO(args.name_model)  #
# results = model.train(data="datasets/iam1-scratch/data.yaml", name="train_iam1-scratch")

# https://docs.wandb.ai/guides/integrations/ultralytics/
# Add W&B callback for Ultralytics
# add_wandb_callback(model, enable_model_checkpointing=True)

date = datetime.datetime.now()
date = str(date)  # '2024-10-15 14:46:33.078424'
date = date.replace(" ", "_")
date = date.replace(":", "_")
date = date.replace(".", "_")
# date_split = str.split(date, " ")
#date = date_split[0] + "_" + date_split[1]  # remove space
project = "character_spotting"

print("Expe name:")
print(date)
# model = YOLO("yolov8x.yaml")
# results = model.train(data="datasets/iam/data.yaml", epochs=600, patience=100, imgsz=512, augment=True, degrees=4,
# iou=0.2, conf=0.1, agnostic_nms=True)
verbose_yolo = False

if args.verbose_yolo == 1:
    verbose_yolo = True

try:
    # name=args.name_exp,
    # Project = project name wandb
    results = model.train(data=args.dataset_yaml,
                          imgsz=(args.img_height_yolo, args.img_weight_yolo),
                          project=project,
                          name=date,
                          batch=args.batch_size,
                          epochs=args.nb_epochs_max,
                          patience=args.patience_early_stopping,
                          verbose=verbose_yolo,
                          degrees=args.degrees,
                          shear=args.shear,
                          translate=args.translate,
                          flipud = 0,
                          fliplr = 0,
                        #   mosaic = 0,
                        #   perspective = 0.001,
                          device = args.gpu_device)
    print(results)
except Exception as e:
    print(e)
####################################################### GRID Search and Testing ##############################################
save_dir = os.path.join(project, date, "weights")

# Finish the W&B run
# wandb.finish()

print("End of training")
path_model = os.path.join(save_dir, "best.pt")
print(path_model)
model = YOLO(path_model)

# grid search param iou, conf on validation
agnostic_nms = False

if args.agnostic_nms == 1:
    agnostic_nms = True

best_cer_val = 1.0
best_iou = 0.35
best_conf = 0.2

print("Grid search on validation split with best model:")
for iou_v in [x / 100.0 for x in range(20, 40, 5)]:
    for conf_v in [x / 100.0 for x in range(10, 30, 5)]:
        print("iou: " + str(iou_v))
        print("conf_v: " + str(conf_v))

        predicted, labels, counter_sub_segmentation, counter_over_segmentation = test_results_v2(data_yaml["val"],
                                                                                                 args.dir_label_val,
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

        if cer < best_cer_val:
            best_cer_val = cer
            best_iou = iou_v
            best_conf = conf_v

print("Evaluation test set")
print("iou: " + str(best_iou))
print("conf_v: " + str(best_conf))

#######################################################################################################################################
list_cer_best_c = []
list_wer_best_c = []

list_db_test_c = []

for path_test in data_yaml["test"]:
    print(path_test)
    list_db_test_c.append(path_test)
    predicted, labels, counter_sub_segmentation, counter_over_segmentation = test_results_v2(path_test,
                                                                                             args.dir_label_test_clean,
                                                                                             model,
                                                                                             args.img_height_yolo,
                                                                                             args.img_weight_yolo,
                                                                                             best_iou,
                                                                                             best_conf,
                                                                                             agnostic_nms)

    cer = jiwer.cer(labels, predicted)
    wer = jiwer.wer(labels, predicted)

    list_cer_best_c.append(cer)
    list_wer_best_c.append(wer)

    print("Best model:")
    print(f"CER: {cer}, WER: {wer}")
    print(f"counter_sub_segmentation: {counter_sub_segmentation}, counter_over_segmentation: {counter_over_segmentation}")

    #wandb.log({"model/best_cer_test": cer})
    #wandb.log({"model/best_wer_test": wer})

    print()


print(list_db_test_c)
print("list_cer_best")
print(list_cer_best_c)
print("list_wer_best")
print(list_wer_best_c)

print()


print("End")

df_menu = ["Trained_on","Score","Model_ID","CLEAN","MIXED","SINGLE_LINE","DOUBLE_LINE","DIAGONAL","CROSS","WAVE","ZIG_ZAG","SCRATCH"]

df_data = [
        [args.trained_on,"CER",args.model_id] + list_cer_best_c,
        [args.trained_on,"WER",args.model_id] + list_wer_best_c
    ]

df = pd.DataFrame(df_data, columns= df_menu)

df.to_csv(f"results_{args.model_id}.csv", index=False)