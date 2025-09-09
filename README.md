# HTR-CharacterSpotting

**HTR-CharacterSpotting** is a Handwritten Text Recognition (HTR) framework that use **YOLOv8** for character detection and utilize autonomous tagging methods to enable character-level ground truth generation.  

## Acknowledgment

This work adopts the **autonomous tagging method** proposed by:  

Majid, Nishatul and Barney Smith, Elisa H.  
*Character spotting and autonomous tagging: offline handwriting recognition for Bangla, Korean and other alphabetic scripts*.  
International Journal on Document Analysis and Recognition (IJDAR), 25(4):245–263, 2022.  
[https://doi.org/10.1007/s10032-022-00397-6](https://doi.org/10.1007/s10032-022-00397-6)

## How to Train
The training scripts are located in the /scripts folder.

Example:
```
./scripts/train_mixed.sh
```

The data folder (cross-out type) should be specified in the `--trained_on` parameter of the training script.

The configuration file (e.g., configuration/crossout_types_mixed.yaml) specifies the data folders for training, validation, and testing.
Since we use YOLOv8, the folder structure should follow the YOLO format as shown below:
```
train
---MIXED_100
   ---images
   ---labels
```
The labels should be per image, following the YOLO format.

## Our Usage and Contribution

We replaced the object detector in [Character spotting](https://doi.org/10.1007/s10032-022-00397-6) with YOLOv8
