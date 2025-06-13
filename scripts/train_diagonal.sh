#!/bin/bash
#SBATCH -A Berzelius-2025-71
#SBATCH -o /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_gapat/logs/multiscripts/%j.err
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -c 4                           # one CPU core
#SBATCH -t 2-12:00:00
#SBATCH --mem=40G

#Experiment ID
EXPERIMENT_ID="53"

# mamba init bash
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate pytorch25

# Parameters
file=train_yolo2.py
config=crossout_types_diagonal.yaml

root=/home/x_gapat/PROJECTS/codes/HTR-CharacterSpotting_YOLOv8
main_script="${root}/src/train/${file}"
config_file="${root}/configuration/${config}"

echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d_%H-%M"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="/home/x_gapat/PROJECTS/logs/HTR-CharacterSpotting_YOLOv8/${EXPERIMENT_ID}_${CURRENTDATE}_ID_${SLURM_JOB_ID}/"
echo "path log :"
echo ${PATHLOG}
mkdir ${PATHLOG}

output_file="${PATHLOG}/log.txt"

export PYTHONPATH=/proj/document_analysis/users/x_gapat/codes/HTR-CharacterSpotting_YOLOv8/

# The job
python -u $main_script \
	"$config_file" \
	"/home/x_gapat/PROJECTS/DATASETS/IAM/YOLO/val/clean/labels" \
	"/home/x_gapat/PROJECTS/DATASETS/IAM/YOLO/test/clean/labels" \
	$PATHLOG \
	--batch_size 16 \
	--nb_epochs_max 200 \
	--patience_early_stopping 50 \
	--img_height_yolo 512 \
	--img_weight_yolo 512 \
	--name_model "yolov8x.yaml" \
	--degrees 4 \
	--gpu_device "cuda" \
	--trained_on "DIAGONAL" \
	--project "$PATHLOG" \
	--model_id "$EXPERIMENT_ID"  \
>> $output_file