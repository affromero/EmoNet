#!/usr/bin/env bash
#USE: ./main.sh -AU 1 -gpu 0 -fold 0
while [[ $# -gt 1 ]]
do
key="$1"
case $key in
    -mode_data|--mode_data)
    mode="$2"
    shift # past argument
    ;;       
    -gpu|--gpu|-GPU|--GPU)
    gpu_id="$2"
    shift # past argument
    ;;    
    -fold|--fold)
    declare -a fold=( "$2" )
    shift # past argument
    ;;    
    -GRAY|--GRAY)
    GRAY=true
    shift # past argument
    ;;    
    -BLUR|--BLUR)
    BLUR=true
    shift # past argument
    ;;              
    *)
esac
shift # past argument or value
done

if [ -z ${mode_data+x} ]; then mode_data="normal"; fi
if [ -z ${fold+x} ]; then declare -a fold=( 0 1 2 all ); fi
if [ -z ${GRAY+x} ]; then GRAY=false; fi
if [ -z ${BLUR+x} ]; then BLUR=false; fi
  for _fold in "${fold[@]}"
  do
    command_train="./main.py -- --GPU=$gpu_id --mode_data=$mode_data --fold=$_fold"
    if [ "$GRAY" = true ]; then command_train+=" --GRAY"; fi  
    if [ "$BLUR" = true ]; then command_train+=" --BLUR"; fi  
    echo $command_train
    eval $command_train
  done