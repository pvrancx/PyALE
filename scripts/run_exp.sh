#!/bin/sh
#set various paths
SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
BASEDIR="${SCRIPTDIR}/.."
RLDIR="${BASEDIR}/src"
LOGDIR="${BASEDIR}/logs"
ROMDIR="${BASEDIR}/roms"
PYTHON="$HOME/anaconda/bin/python"

#Experimental configuration
EXP_NAME="sarsa_RAM"
EXPERIMENT="exp/generic_experiment.py"
EXPERIMENT_OPTIONS="--maxsteps 2000  --numeps 100 --numtrials 3"
AGENT="agents/ALESarsaAgent.py"
AGENT_OPTIONS='--eps 0.05 --lambda 0.5 --alpha 0.1 --features RAM --actions 0 1 3 4'
ALE_OPTIONS="-game_controller rlglue  -frame_skip 30 -repeat_action_probability 0.0"
GAME="space_invaders.bin"

echo "Writing logs to ${LOGDIR}"

#######Imports#################
export PYTHONPATH=$RLDIR:$PYTHONPATH
export RLGLUE_PORT=1028
###############################

mkdir -p "$LOGDIR/$EXP_NAME"

cd $RLDIR
#start rlglue
rl_glue&
#run agent
$PYTHON $AGENT $AGENT_OPTIONS --savepath "$LOGDIR/$EXP_NAME" > $LOGDIR/agent-$EXP_NAME.log 2>&1 &
#run experiment
$PYTHON $EXPERIMENT $EXPERIMENT_OPTIONS >> $LOGDIR/exp-$EXP_NAME.log 2>&1 &
#run environment (no '&' or job quits!)
ale $ALE_OPTIONS $ROMDIR/$GAME > $LOGDIR/ale-$EXP_NAME.log 2>&1
