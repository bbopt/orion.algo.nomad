import os
import argparse
import logging
import sys
import json

from orion.client import report_objective

def bb(var):
    x = var['x']
    y = var['y']
    return 100*(y-x*x)**2+(1-x)**2


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config',default='./initial_point.json')
    
    args, _ = parser.parse_known_args()

     
    # Open json file where the variables values are stored
    with open(vars(args)['config']) as json_file:
        variables = json.load(json_file)

        try:
            f = bb(variables)
            report_objective(f)     # Return objective to Orion
            print(variables['x'],variables['y'],f)
            
        except Exception as exception:
            raise
