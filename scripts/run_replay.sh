#!/bin/bash

#Experimental configuration
RANDOM_SEED=${RANDOM_SEED:-"42"}
DISABLE_TRACES=${DISABLE_TRACES:-""}
REPLAY_TIMES=${REPLAY_TIMES:-3}


EXP_NAME="replay_sarsa"
EXPERIMENT="exp/generic_experiment.py"
EXPERIMENT_OPTIONS="--maxsteps 2000  --numeps 3000 --numtrials 5"
AGENT="agents/ALEReplayAgent.py"
AGENT_REPLAY_OPTIONS="--random_seed=${RANDOM_SEED} --replay_memory=5000 --replay_times=${REPLAY_TIMES} --replay_frequency=100"
AGENT_OPTIONS="--eps 0.05 --lambda 0.5 --alpha 0.1 --actions 0 1 3 4 ${AGENT_REPLAY_OPTIONS}"
ALE_OPTIONS="-game_controller rlglue  -frame_skip 30 -repeat_action_probability 0.0 -random_seed ${RANDOM_SEED}"
GAME="space_invaders"

# Boolean flags are handled a bit more annoyingly
if [ $DISABLE_TRACES ]; then 
  AGENT_OPTIONS="${AGENT_OPTIONS} --disable-traces"
fi

# This will take care of the rest of the configuration and actual execution
source common.sh

