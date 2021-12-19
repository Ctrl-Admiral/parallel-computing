#!/bin/bash

#SBATCH --job-name=run.sbatch              # Название задачи
#SBATCH --error=run.err                    # Файл для вывода ошибок
#SBATCH --output=run.log                   # Файл для вывода результатов
#SBATCH --time=00:05:00                   # Максимальное время выполнения
#SBATCH --ntasks=64                       # Количество MPI процессов
#SBATCH --ntasks-per-node=16              # Количество процессов на каждом узле
#SBATCH --nodes=4                         # Требуемое кол-во узлов
#SBATCH --gpus=0                          # Требуемое кол-во GPU

module load INTEL/parallel_studio_xe_2020_u4_ce
module load openmpi/4.0.5
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
srun /home/sigavarina/Projects/parallel-computing/task2
