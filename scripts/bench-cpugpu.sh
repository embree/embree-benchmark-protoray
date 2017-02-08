if [ "$#" -ne 1 ]
then
    echo "Usage: $0 (embree|optix)"
    exit 1
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

protoray="./protoray"
result="bench-cpugpu-$1"
scenes="mazda villa artdeco powerplant sanmiguel"
resolution=3840,2160
spp=64

line="scene,renderMray,buildMprim"
echo "$line" > $result.csv

export LD_LIBRARY_PATH=".:$LD_LIBRARY_PATH"

for scene in $scenes
do
    sleep 60
    $protoray render $scene.mesh -no-mtl -dev $device -r $renderer -a $accel -no-sbvh -sampler random -size $resolution -spp $spp -do -benchmark $result-$scene
    source $result-$scene.txt
    line="$scene,$stats_mray,$stats_buildMprim"
    echo "$line" >> $result.csv
done
