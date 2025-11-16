# SLURM Usage

SLURM is a job scheduler for Linux clusters. It is used to manage the allocation and execution of resources on a cluster.

The HPC cluster at the University of Potsdam uses SLURM.

## Quick facts

-   Jobs are submitted with `sbatch`, interactive sessions started via `srun`.
-   Your home directory is in `/home/username`, your scratch directory in `/work/username`.
    -   Only `/home` is backed up centrally (snapshots of last 14 days).
    -   It is your own responsibility to take backups of important data in your scratch directory.
    -   The scratch filesystem `/work` should be used for compute jobs (not `/home`!).
-   The HPC-cluster is a shared resource, so please be considerate (especially regarding usage of storage space in `/work`).

## Starting an Interactive SLURM Session

Starting an interactive SLURM session gives a shell on a GPU node. As the default his is necessary to run any scripts that require a GPU.

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=2 --mem=24G --pty bash
```

This gives a shell on a GPU node.

In that new shell:

```bash
cd ~/masters-thesis
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
#SBATCH --partition=gpu # necessary for GPU jobs
#SBATCH --gres=gpu:1 # number of GPUs per task
#SBATCH --nodelist=n-hpc-gz6 # replace with the node you want to use, if you want to use a specific node
#SBATCH --ntasks=1 # number of tasks per node
#SBATCH --cpus-per-task=4 # number of CPUs per task
#SBATCH --mem=40G # maximum memory per task --> process will be killed if it exceeds this limit!
#SBATCH --chdir=/work/utsch/masters-thesis # set the working directory
#SBATCH --output=slurm-hrpo-gsm8k-%j.out # set the output file

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
cd /work/utsch/jobs
sbatch hrpo_gsm8k.job
```

Check the job status:

```bash
squeue -u $USER
```

Check the job output:

```bash
cd /work/utsch/jobs/masters-thesis
cat hrpo_gsm8k.job.out
```

To follow logs in real-time as the job runs:

```bash
tail -f /work/utsch/masters-thesis/slurm-hrpo-gsm8k-<JOB_ID>.out
```
