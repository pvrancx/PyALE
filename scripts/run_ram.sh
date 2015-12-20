#!/bin/sh
#set various paths
SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
BASEDIR="${SCRIPTDIR}/.."
RLDIR="${BASEDIR}/src"
LOGDIR="${BASEDIR}/logs_ram"
ROMDIR="${BASEDIR}/roms"
PYTHON=${PYTHON:-"python"}

#Experimental configuration
EXP_NAME="sarsa_RAM"
EXPERIMENT="exp/generic_experiment.py"
EXPERIMENT_OPTIONS="--maxsteps 2000  --numeps 3000 --numtrials 5"
AGENT="agents/ALESarsaAgent.py"
AGENT_OPTIONS='--eps 0.05 --lambda 0.5 --alpha 0.1 --features RAM --actions 0 1 3 4'
ALE_OPTIONS="-game_controller rlglue  -frame_skip 30 -repeat_action_probability 0.0"
GAME="space_invaders.bin"

echo "Using python binary ${PYTHON}"
echo "Writing logs to ${LOGDIR}"

####### Other Settings ########
export PYTHONPATH=$RLDIR:$PYTHONPATH
# This one didn't work. Maybe the socket didn't close in time? RLGlue refused to take this port
# export RLGLUE_PORT=`python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'`

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

cd $RLDIR
#start rlglue
rl_glue&
#run agent
$PYTHON $AGENT $AGENT_OPTIONS --savepath "$LOGDIR/$EXP_NAME" > $LOGDIR/agent-$EXP_NAME.log 2>&1 &
#run experiment
$PYTHON $EXPERIMENT $EXPERIMENT_OPTIONS >> $LOGDIR/exp-$EXP_NAME.log 2>&1 &
#run environment (no '&' or job quits!)
ale $ALE_OPTIONS $ROMDIR/$GAME > $LOGDIR/ale-$EXP_NAME.log 2>&1
