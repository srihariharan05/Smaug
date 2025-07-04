#!/usr/bin/env bash

. ./model_files

cfg_home=`pwd`
gem5_dir=${ALADDIN_HOME}/../..
bmk_dir=`git rev-parse --show-toplevel`/../build/bin

${gem5_dir}/build/X86/gem5_dma_uncache.opt \
  --debug-flags=Aladdin,HybridDatapath \
  --outdir=${cfg_home}/outputs_dma_uncache_nostatdump \
  --stats-db-file=stats.db \
  -r -e \
  ${gem5_dir}/configs/aladdin/aladdin_se.py \
  --num-cpus=1 \
  --mem-size=64GB \
  --mem-type=LPDDR4_3200_2x16  \
  --sys-clock=1GHz \
  --cpu-clock=2GHz \
  --cpu-type=DerivO3CPU \
  --caches \
  --l2cache \
  --enable_prefetchers \
  --l1d_size=128kB \
  --l1i_size=256kB \
  --l2_size=16MB \
  --l1d-hwp-type=StridePrefetcher \
  --l2-hwp-type=SignaturePathPrefetcherV2 \
  --accel_cfg_file=gem5_dma.cfg \
  --fast-forward=10000000000 \
  -c ${bmk_dir}/smaug \
  -o "minerva_smv_topo_dma.pbtxt minerva_smv_params.pb  --gem5 --iterations=10 --debug-level=0 --num-accels=1" \
  #> stdout 2> stderr


#  --l2_size=2097152 \
#  --l2_assoc=16 \
#  --cacheline_size=32 \
