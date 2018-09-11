#!/bin/bash

protoray="./protoray"
resultsDir="results"
scenesDir="scenes"
if [ ! -d "$scenesDir" ]; then
  scenesDir="."
fi
scenes="mazda villa artdeco powerplant sanmiguel"

resolution=3840,2160
spp=64

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ] && [ "$#" -ne 3 ]
then
    echo "Usage: $0 [protoray_path] (embree|optix) [scene]"
    exit 1
fi

if [[ "$1" == *protoray* ]]
then
  protoray="$1"
  shift
fi

if [ "$1" == "embree" ]
then
    device="cpu"
    accel="embree"
    renderer="diffusePacketFast"
elif [ "$1" == "optix" ]
then
    device="cuda"
    accel="optix"
    renderer="diffuseFast"
else
    echo "Invalid argument: $1"
    exit 1
fi

if [ "$#" -eq 2 ]
then
    scenes="$2"
fi

result="$resultsDir/bench-cpugpu-$1"
mkdir -p $resultsDir

line="scene,renderMray,buildMprim"
echo "$line" > $result.csv

export LD_LIBRARY_PATH="`dirname $protoray`:lib:/opt/intel/lib/intel64:$LD_LIBRARY_PATH"

for scene in $scenes
do
    #sleep 60
    $protoray render "$scenesDir/$scene.mesh" -no-mtl -dev $device -r $renderer -maxDepth 6 -a $accel -sampler random -size $resolution -spp $spp -do -benchmark $result-$scene
    source $result-$scene.txt
    line="$scene,$stats_mray,$stats_buildMprim"
    echo "$line" >> $result.csv
done

echo
cat $result.csv
