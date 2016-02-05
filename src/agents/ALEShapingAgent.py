# -*- coding: utf-8 -*-
import argparse
import copy

from rlglue.agent import AgentLoader as AgentLoader

from util.ALEFeatures import BasicALEFeatures
from agents.ALESarsaAgent import ALESarsaAgent

import numpy as np

class ALEShapingAgent(ALESarsaAgent):

    @classmethod
    def register_with_parser(cls, parser):
        super(ALEShapingAgent, cls).register_with_parser(parser)

    
    def __init__(self, args):
        super(ALEShapingAgent, self).__init__(args)
