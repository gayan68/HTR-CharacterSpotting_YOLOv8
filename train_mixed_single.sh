#!/bin/bash
#SBATCH -o /proj/document_analysis/users/x_scorb/logs/handwriting_crossout/%j.out
#SBATCH -e /proj/document_analysis/users/x_scorb/logs/handwriting_crossout/%j.err
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -c 4                           # one CPU core
#SBATCH -t 3-0:00:00
#SBATCH --mem=40G

# conda init bash
# source /home/x_scorb/miniconda3/etc/profile.d/conda.sh
# conda activate htr-yolo

# Parameters
main_script=src/train/train_yolo2.py
# config_file=/proj/document_analysis/users/x_scorb/codes/handwriting_crossout/configuration/config_gpu_iam.json

echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d_%H-%M"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="logs/${CURRENTDATE}/"
echo "path log :"
echo ${PATHLOG}
mkdir ${PATHLOG}

echo "GPU id: 1"
#echo ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}


output_file="${PATHLOG}/log.txt"

# export PYTHONPATH=/proj/document_analysis/users/x_scorb/codes/handwriting_crossout/

# The job
python -u $main_script \
	"configuration/crossout_types_single_line.yaml" \
	"/home/gayapath/PROJECTS/DATASETS/IAM_local_DGX_2/YOLO/val/mixed/labels" \
	"/home/gayapath/PROJECTS/DATASETS/IAM_local_DGX_2/YOLO/test/clean/labels" \
	$PATHLOG \
	--batch_size 16 \
	--nb_epochs_max 200 \
	--patience_early_stopping 50 \
	--img_height_yolo 512 \
	--img_weight_yolo 512 \
	--name_model "yolov8x.yaml" \
	--degrees 4 \
	--gpu_device "cuda:1" \
	--trained_on "SINGLE_LINE" \
	--model_id "35" \
>> $output_file


