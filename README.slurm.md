# SLURM Usage

## Starting an Interactive SLURM Session

Starting an interactive SLURM session gives a shell on a GPU node. As the default his is necessary to run any scripts that require a GPU.

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=2 --mem=24G --time=0-01:00 --pty bash
```

This gives a shell on a GPU node.

In that new shell:

```bash
source .venv/bin/activate
export http_proxy=http://proxy2.uni-potsdam.de:3128
export https_proxy=http://proxy2.uni-potsdam.de:3128
python src/hrpo/hrpo_gsm8k.py
```

## Creating a new SLURM Batch Job

```bash
cd $SCRATCH/jobs
nano hrpo_gsm8k.job
```

Add the following content to the file:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --chdir=/work/utsch/masters-thesis
#SBATCH --output=slurm-hrpo-gsm8k-%j.out

echo "Running on $(hostname) at $(date)"

# load modules if your cluster uses them, e.g.:
# module load cuda/12.x
# module load python/3.11

# set proxies for internet access of node
export http_proxy=http://proxy2.uni-potsdam.de:3128
export https_proxy=http://proxy2.uni-potsdam.de:3128

# activate your env
source ~/masters-thesis/.venv/bin/activate

# run HRPO script
python ~/masters-thesis/src/hrpo/hrpo_gsm8k.py
```

## Queueing a SLURM Batch Job

Queue the job:

```bash
cd $SCRATCH/jobs
sbatch hrpo_gsm8k.job
```

Check the job status:

```bash
squeue -u $USER
```

Check the job output:

```bash
cd $SCRATCH/jobs/masters-thesis
cat hrpo_gsm8k.job.out
```
