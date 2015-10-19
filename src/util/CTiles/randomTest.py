# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 09:39:37 2015

@author: pvrancx
"""
import argparse

# tile timing tests

def runTest(seed = 65597):

    import tiles

    i=5
    j=6
    print 'seed: '+str(seed)
    tiles.setseed(seed)
    print tiles.tiles(10, 2048, [i*0.5, j*0.5])
            

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='run tiles test')
    parser.add_argument('--seed', metavar='S', type=int, default=65597,
                    help='seed')


    args = parser.parse_args()
    runTest(args.seed)