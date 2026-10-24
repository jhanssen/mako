#!/bin/zsh
# Memory stress test: decode with the expert arena while another process writes a large file (dirty page cache), all
# under mem_watchdog, which kills both if available memory or the compressor crosses its limits. Stages ramp up;
# the script stops at the first stage the watchdog had to kill. Logs (fsync'ed) go to ~/mako-stress-logs.
#   bench/stress_write.sh <model shard 1> [first stage] [last stage]
# Stages: 1: 30 GB arena, 40 GB written; 2: default arena, 40 GB; 3: default arena, 150 GB (more than RAM).
set -u
MODEL=$1
FIRST=${2:-1}
LAST=${3:-3}
ROOT=${0:A:h:h}
LOGS=~/mako-stress-logs
TMPDIR_W=~/mako-stress-tmp
mkdir -p $LOGS $TMPDIR_W

stage() {
    local n=$1 arena=$2 write_gb=$3
    local log=$LOGS/stage$n.log
    echo "stage $n: arena ${arena:-default}, writing ${write_gb} GB" | tee -a $log
    local arena_opt=()
    [[ -n $arena ]] && arena_opt=(--arena-gb $arena)
    $ROOT/build/bench/mem_watchdog --log $log --min-avail-gb 6 --max-compressed-gb 4 -- /bin/zsh -c "
        $ROOT/build/tools/mako_run $MODEL 785,6722,315 -n 1500 --top 1 ${arena_opt[*]} --stats summary > $LOGS/stage$n.run.log 2>&1 &
        RUN=\$!
        sleep 15
        dd if=/dev/zero of=$TMPDIR_W/fill bs=8m count=$((write_gb * 128)) > $LOGS/stage$n.dd.log 2>&1
        wait \$RUN"
    local rc=$?
    rm -f $TMPDIR_W/fill
    echo "stage $n: exit $rc" | tee -a $log
    return $rc
}

for n in {$FIRST..$LAST}; do
    case $n in
    1) stage 1 30 40 ;;
    2) stage 2 "" 40 ;;
    3) stage 3 "" 150 ;;
    esac || { echo "stopped after stage $n (watchdog or failure); see $LOGS"; exit 1; }
done
echo "all stages passed"
