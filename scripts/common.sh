#!/bin/bash
# This script is never meant to be executed directly

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
BASEDIR="${SCRIPTDIR}/.."
RLDIR="${BASEDIR}/src"
ROMDIR="${BASEDIR}/roms"
# Can be overridden, for example by calling the script as:
# PYTHON=~/anaconda/bin/python ./run_basic.sh
PYTHON=${PYTHON:-"python"}

# Every experiment gets a 'unique' directory, so no accidental overwriting occurs
TIME_STR=`python -c "import time; print time.strftime('%d-%m-%Y_%H-%M-%S')"`
LOGDIR="${BASEDIR}/logs/${EXP_NAME}_${GAME}_${TIME_STR}"
echo "Using python binary ${PYTHON}"
echo "Writing logs to ${LOGDIR}"

# Python profiling can be enabled here
ENABLE_PROFILER=0
PROFILE_STRING=""
if [ $ENABLE_PROFILER -ne 0 ]; then
  PROFILE_STRING="-m cProfile -o '$LOGDIR/profile'"
fi

####### Other Settings ########
export PYTHONPATH=$RLDIR:$PYTHONPATH
# Find an open port starting from 4096
for port in $(seq 4096 65000); do echo -ne "\035" | telnet 127.0.0.1 $port > /dev/null 2>&1; [ $? -eq 1 ] && export RLGLUE_PORT=$port && break; done
echo "Found RLGlue port $RLGLUE_PORT"

# Make sure Python doesn't buffer output
export PYTHONUNBUFFERED="YEAP"
###############################

# If anything goes wrong or script is killed, kill all subprocesses too
# Kills background jobs only. All the jobs below are background jobs.
# The profiler won't write results with this trap
if [ $ENABLE_PROFILER -eq 0 ]; then
  trap 'kill $(jobs -p)' EXIT
else
  echo "Disabling trap because profiler is running"
fi
# More elaborate. Kill thoroughly.
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# This directory will holds results
mkdir -p "$LOGDIR/results"

# Keep a handy simlink to the latest results for debugging purposes
ln -sfT $LOGDIR "${BASEDIR}/logs/last_${EXP_NAME}_${GAME}"

cd $RLDIR
#start rlglue
rl_glue&
#run agent
$PYTHON $PROFILE_STRING $AGENT $AGENT_OPTIONS --savepath "$LOGDIR/results" > $LOGDIR/agent.log 2>&1 &
#run experiment
$PYTHON $EXPERIMENT $EXPERIMENT_OPTIONS >> $LOGDIR/experiment.log 2>&1 &
#run environment (no '&' or job quits!)
ale $ALE_OPTIONS $ROMDIR/"${GAME}.bin" > $LOGDIR/ale.log 2>&1
