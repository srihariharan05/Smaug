#!usr/bin/env bash

#export SMAUG_HOME=/mnt/research/Gratz_Paul_V/Students/Sriram_Sri_Hariharan/smaug
#export ALADDIN_HOME=/mnt/research/Gratz_Paul_V/Students/Sriram_Sri_Hariharan/smaug/gem5-aladdin/src/aladdin
#export PATH=$PATH:/mnt/research/Gratz_Paul_V/Students/Sriram_Sri_Hariharan/gdb_bin/
#cd $SMAUG_HOME/experiments/models/MobileDet/

cfg_home=`pwd`
gem5_dir=${ALADDIN_HOME}/../../
bmk_dir=${SMAUG_HOME}/build/bin
spec_dir=${SMAUG_HOME}/benchmarks/spec_cpu2006

${gem5_dir}/build/X86/gem5.opt \
  --outdir=${cfg_home}/outputs_SPRP_lg_hmmr \
  --stats-db-file=stats.db \
  -r -e \
  ${gem5_dir}/configs/aladdin/aladdin_se.py \
  --num-cpus=2 \
  --multicluster \
  --mem-size=64GB \
  --mem-type=LPDDR4_3200_2x16 \
  --mem-ranks=8 \
  --mem-devices=8 \
  --sys-clock=1GHz \
  --cpu-clock=2GHz \
  --cpu-type=DerivO3CPU \
  --caches \
  --l2cache \
  --l1d_size=128kB \
  --l1i_size=256kB \
  --l2_size=2MB \
  --l3_size=1MB \
  --enable_prefetchers \
  --l1d-hwp-type=StridePrefetcher \
  --l2-hwp-type=StridePrefetcher \
  --l3-hwp-type=TaggedPrefetcher \
  --l2-hwp-degree=8 \
  --l3-hwp-degree=48 \
  --snoop-filter-size=32MB \
  --l3-repl-policy=SPRP \
  --acc-wr-promote \
  --mod-rrpv \
  --rrip-hp \
  --rrpv-bits=3 \
  --fast-forward=10000000000 \
  --accel_cfg_file=gem5.cfg \
  -c ${bmk_dir}/smaug_512 \
  -o "edsr_smv_topo.pbtxt edsr_smv_params.pb --sample-num=1 --sample-level=very_high --gem5=true --num-accels=1 --debug-level=0 --iterations=1" \
  #-c ${bmk_dir}/smaug_512;${spec_dir}/bin/cactusADM \
  #-o mobiledet_smv_topo.pbtxt mobiledet_smv_params.pb --sample-num=1 --sample-level=very_high --gem5=true --num-accels=1 --debug-level=0 --iterations=1;${spec_dir}/data/cactusADM/benchADM.par \
  #-c ${bmk_dir}/smaug_512 \
  #-o "mobiledet_smv_topo.pbtxt mobiledet_smv_params.pb --sample-num=1 --sample-level=very_high --gem5=true --num-accels=1 --debug-level=0 --iterations=1" \
#; gcc-smaller.c -O5 -fipa-pta -o gcc-smaller.opts-O3_-fipa-pta.s" \

 # -c ${bmk_dir}/smaug_512;${spec_dir}/bin/gcc \
 # -o "mobiledet_smv_topo.pbtxt mobiledet_smv_params.pb --sample-num=1 --sample-level=very_high --gem5=true --num-accels=1 --debug-level=0 --iterations=1;${spec_dir}/data/gcc/expr.ini -o ${spec_dir}/data/gcc/expr.s" \
 # -c ../SPEC_17/541.leela_r/base_refrate/leela_r_base.x86-64  \
 # -o ../SPEC_17/541.leela_r/base_refrate/ref.sgf \
 # --enable_prefetchers \
 # --l1d-hwp-type=StridePrefetcher \
 # --l2-hwp-type=StridePrefetcher \
 # --l3-hwp-type=TaggedPrefetcher \
 # --l2-hwp-degree=8 \
 # --l3-hwp-degree=24 \
