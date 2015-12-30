#!/bin/bash
#set various paths
SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
BASEDIR="${SCRIPTDIR}/.."
RLDIR="${BASEDIR}/src"
ROMDIR="${BASEDIR}/roms"
PYTHON=${PYTHON:-"python"}

# Every experiment gets a 'unique' directory, so no accidental overwriting occurs
TIME_STR=`python -c "import time; print time.strftime('%d-%m-%Y_%H-%M')"`
LOGDIR="${BASEDIR}/logs/${EXP_NAME}_${GAME}_${TIME_STR}"
echo "Using python binary ${PYTHON}"
echo "Writing logs to ${LOGDIR}"

####### Other Settings ########
# This one didn't work. Maybe the socket didn't close in time? RLGlue refused to take this port
# export RLGLUE_PORT=`python -c 'import socket; s=socket.socket(); s.bind(("",
# 0)); print(s.getsockname()[1]); s.close()'`
export PYTHONPATH=$RLDIR:$PYTHONPATH
# Find an open port starting from 4096
for port in $(seq 4096 65000); do echo -ne "\035" | telnet 127.0.0.1 $port > /dev/null 2>&1; [ $? -eq 1 ] && export RLGLUE_PORT=$port && break; done
echo "Found RLGlue port $RLGLUE_PORT"

export PYTHONUNBUFFERED="YEAP"
###############################

mkdir -p "$LOGDIR/$EXP_NAME"

# If anything goes wrong or script is killed, kill all subprocesses too
# Kills background jobs only. All the jobs below are background jobs.
trap 'kill $(jobs -p)' EXIT
# More elaborate. Kill thoroughly.
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# Keep a handy simlink to the latest results for debugging purposes
ln -sfT $LOGDIR "last_${EXP_NAME}_${GAME}"

cd $RLDIR
#start rlglue
rl_glue&
#run agent
$PYTHON $AGENT $AGENT_OPTIONS --savepath "$LOGDIR/results" > $LOGDIR/agent.log 2>&1 &
#run experiment
$PYTHON $EXPERIMENT $EXPERIMENT_OPTIONS >> $LOGDIR/experiment.log 2>&1 &
#run environment (no '&' or job quits!)
ale $ALE_OPTIONS $ROMDIR/"${GAME}.bin" > $LOGDIR/ale.log 2>&1
