#!/bin/bash

#Experimental configuration
RANDOM_SEED=${RANDOM_SEED:-"42"}
DISABLE_TRACES=${DISABLE_TRACES:-""}

EXP_NAME="sarsa_BASIC"
EXPERIMENT="exp/generic_experiment.py"
EXPERIMENT_OPTIONS="--maxsteps 2000  --numeps 3000 --numtrials 1"
AGENT="agents/ALESarsaAgent.py"
AGENT_OPTIONS="--eps 0.05 --lambda 0.5 --alpha 0.1 --actions 0 1 3 4 --features RAM"
ALE_OPTIONS="-game_controller rlglue  -frame_skip 30 -repeat_action_probability 0.0 -random_seed ${RANDOM_SEED}"
GAME="space_invaders"

# Boolean flags are handled a bit more annoyingly
if [ $DISABLE_TRACES ]; then 
  AGENT_OPTIONS="${AGENT_OPTIONS} --disable-traces"
fi

# This will take care of the rest of the configuration and actual execution
source common.sh
