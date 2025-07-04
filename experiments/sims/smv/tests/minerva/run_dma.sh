#!/usr/bin/env bash

. ./model_files

cfg_home=`pwd`
gem5_dir=${ALADDIN_HOME}/../..
#bmk_dir=`git rev-parse --show-toplevel`/../build/bin
bmk_bin=${SMAUG_HOME}/benchmarks/spec_cpu2006/bin
bmk_data=${SMAUG_HOME}/benchmarks/spec_cpu2006/data

${gem5_dir}/build/X86/gem5.opt \
  --outdir=${cfg_home}/outputs \
  --stats-db-file=stats.db \
  ${gem5_dir}/configs/aladdin/aladdin_se.py \
  --num-cpus=1 \
  --mem-size=4GB \
  --mem-type=LPDDR4_3200_2x16  \
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
  --l1d-hwp-type=StridePrefetcher \
  --l2-hwp-type=StridePrefetcher \
  --l3-hwp-type=TaggedPrefetcher \
  --l3-repl-policy=SPRP \
  --acc-wr-promote \
  --accel_cfg_file=gem5_dma.cfg \
  --fast-forward=10000000000 \
  -c ${bmk_bin}/cactusADM \
  -o "${bmk_data}/cactusADM/benchADM.par" \
  #-o "minerva_smv_topo.pbtxt minerva_smv_params.pb  --gem5=true --iterations=1 --debug-level=0 --num-accels=1" \
  #> stdout 2> stderr 
  # --connect-dma-to-L2 \
  # RRIPRP, BRRIPRP, LRURP, BIPRP,


#  --l2_size=2097152 \
#  --l2_assoc=16 \
#  --cacheline_size=32 \
