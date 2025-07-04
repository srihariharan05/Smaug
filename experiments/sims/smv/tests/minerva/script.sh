#!/usr/bin/env bash

#. ./model_files

cfg_home=`pwd`
gem5_dir=${ALADDIN_HOME}/../..
bmk_dir=`git rev-parse --show-toplevel`/../build/bin

${gem5_dir}/build/X86/gem5.opt \
  --debug-flags=Aladdin,HybridDatapath \
  --outdir=${cfg_home}/outputs \
  --stats-db-file=stats.db \
  ${gem5_dir}/configs/aladdin/aladdin_se.py \
  --num-cpus=1 \
  --mem-size=4GB \
  --mem-type=LPDDR4_3200_2x16  \
  --sys-clock=1.75GHz \
  --cpu-clock=3.53GHz \
  --cpu-type=DerivO3CPU \
  --ruby \
  --access-backing-store \
  --caches \
  --l2cache \
  --enable_prefetchers \
  --l1d_size=96kB \
  --l1i_size=192kB \
  --l2_size=12MB \
  --l3_size=6MB \
  --l1d-hwp-type=StridePrefetcher \
  --l2-hwp-type=SignaturePathPrefetcher \
  --accel_cfg_file=gem5.cfg \
  --fast-forward=10000000000 \
  -c ${bmk_dir}/smaug \
  -o "minerva_smv_topo.pbtxt minerva_smv_params.pb --gem5 --debug-level=0 --num-accels=1" \
  > stdout 2> stderr


#  --l2_size=2097152 \
#  --l2_assoc=16 \
#  --cacheline_size=32 \
