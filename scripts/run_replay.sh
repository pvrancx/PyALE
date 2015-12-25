#!/bin/bash
#set various paths
SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
BASEDIR="${SCRIPTDIR}/.."
RLDIR="${BASEDIR}/src"
ROMDIR="${BASEDIR}/roms"
PYTHON=${PYTHON:-"python"}
RANDOM_SEED=42

#Experimental configuration
EXP_NAME="replay_sarsa"
EXPERIMENT="exp/generic_experiment.py"
EXPERIMENT_OPTIONS="--maxsteps 2000  --numeps 5 --numtrials 2"
AGENT="agents/ALEReplayAgent.py"
AGENT_OPTIONS='--eps 0.05 --lambda 0.5 --alpha 0.1 --actions 0 1 3 4'
AGENT_REPLAY_OPTIONS="--random_seed=${RANDOM_SEED} --replay_memory=1000000
--replay_times=3"
ALE_OPTIONS="-game_controller rlglue  -frame_skip 30 -repeat_action_probability
0.0 -random_seed ${RANDOM_SEED}"
GAME="space_invaders"

# Every experiment gets a 'unique' directory, so no accidental overwriting occurs
TIME_STR=`python -c "import time; print time.strftime('%d-%m-%Y_%H-%M')"`
LOGDIR="${BASEDIR}/logs/${EXP_NAME}_${GAME}_${TIME_STR}"

echo "Using python binary ${PYTHON}"
echo "Writing logs to ${LOGDIR}"

####### Other Settings ########
export PYTHONPATH=$RLDIR:$PYTHONPATH
# This one didn't work. Maybe the socket didn't close in time? RLGlue refused to take this port
# export RLGLUE_PORT=`python -c 'import socket; s=socket.socket(); s.bind(("",
# 0)); print(s.getsockname()[1]); s.close()'`

# Find an open port starting from 4096
for port in $(seq 4096 65000); do echo -ne "\035" | telnet 127.0.0.1 $port > /dev/null 2>&1; [ $? -eq 1 ] && export RLGLUE_PORT=$port && break; done
export PYTHONUNBUFFERED="YEAP"
###############################

echo "Found RLGlue port $RLGLUE_PORT"

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
$PYTHON $AGENT $AGENT_OPTIONS $AGENT_REPLAY_OPTIONS --savepath "$LOGDIR/$EXP_NAME" > $LOGDIR/agent-$EXP_NAME.log 2>&1 &
#run experiment
$PYTHON $EXPERIMENT $EXPERIMENT_OPTIONS >> $LOGDIR/exp-$EXP_NAME.log 2>&1 &
#run environment (no '&' or job quits!)
ale $ALE_OPTIONS $ROMDIR/"${GAME}.bin" > $LOGDIR/ale-$EXP_NAME.log 2>&1
