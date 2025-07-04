export SMAUG_HOME=/home/grads/s/srihariharan05/Thesis/docker_smaug/smaug
export ALADDIN_HOME=/home/grads/s/srihariharan05/Thesis/docker_smaug/smaug/gem5-aladdin/src/aladdin
cd $SMAUG_HOME/experiments/models/MOSAIC/
sh run_script_multi.sh
#$SMAUG_HOME/build/bin/smaug-instrumented_512 mobiledet_smv_topo.pbtxt mobiledet_smv_params.pb --gem5=false --sample-level=very_high --sample-num=1
