#!/bin/sh
#SBATCH --job-name=MOSAIC_olym_home            # Job name
#SBATCH --mail-type=ALL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=srihariharan05@tamu.edu  # Where to send mail
#SBATCH --nodes=1                       # Use one node - Multiple node jobs require MPI
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=2               # Number of CPU cores per task
#SBATCH --time=11:59:59                 # Time limit hrs:min:sec
#SBATCH --output=multi_sbatch.txt        # Standard output and error log
#SBATCH --partition=cpu-research           # Partition/Queue to run in
#SBATCH --qos=olympus-cpu-research #olympus-academic                 # QOS to use
# set woring directory if different than the directory you run the script from
cd ../../../../
singularity exec smaug_latest.sif "`pwd`/smaug/experiments/models/MOSAIC/trace_script.sh"
