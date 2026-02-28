#!/bin/bash -l
##----------------------- Start job description -----------------------
#SBATCH --partition=standard
#SBATCH --job-name=test_fakenews_distilbert_dnn
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=claudio.catalano.leiva@alumnos.upm.es
##------------------------ End job description ------------------------

module purge && module load python3

python3-m pip install --user -r requirements.txt
export TOKENIZERS_PARALLELISM=false

srun python3 -u train_distilbertDNN_fakenews.py

