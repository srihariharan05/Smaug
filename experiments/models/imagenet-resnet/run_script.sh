#!usr/bin/env bash

#export SMAUG_HOME=/mnt/research/Gratz_Paul_V/Students/Sriram_Sri_Hariharan/smaug
#export ALADDIN_HOME=/mnt/research/Gratz_Paul_V/Students/Sriram_Sri_Hariharan/smaug/gem5-aladdin/src/aladdin
#export PATH=$PATH:/mnt/research/Gratz_Paul_V/Students/Sriram_Sri_Hariharan/gdb_bin/
cd $SMAUG_HOME/experiments/models/MobileNetV2/

cfg_home=`pwd`
gem5_dir=${ALADDIN_HOME}/../../
bmk_dir=${SMAUG_HOME}/build/bin/

${gem5_dir}/build/X86/gem5.opt \
  --outdir=${cfg_home}/outputs_No_pf_1_1 \
  -r -e \
  ${gem5_dir}/configs/aladdin/aladdin_se.py \
  --num-cpus=1 \
  --mem-size=64GB \
  --mem-type=LPDDR4_3200_2x16 \
  --mem-ranks=8 \
  --mem-devices=8 \
  --sys-clock=1GHz \
  --cpu-clock=2GHz \
  --cpu-type=DerivO3CPU \
  --caches \
  --l2cache \
  --enable_prefetchers \
  --l1d_size=128kB \
  --l1i_size=256kB \
  --l2_size=8MB \
  --l3_size=4MB \
  --snoop-filter-size=32MB \
  --l1d-hwp-type=StridePrefetcher \
  --l2-hwp-type=StridePrefetcher \
  --l3-hwp-type=SignaturePathPrefetcherV2 \
  --l2-hwp-degree=32 \
  --l3-hwp-degree=48 \
  --l3-repl-policy=RRIPRP \
  --accel_cfg_file=gem5.cfg \
  --fast-forward=10000000000 \
  -c ${bmk_dir}/smaug_512 \
  -o "mobiledet_smv_topo.pbtxt mobiledet_smv_params.pb --sample-num=1 --sample-level=very_high --gem5=true --num-accels=1 --debug-level=0 --iterations=1" \
