#!/usr/bin/env python
import os
import argparse
import logging
import sys
import json

from orion.client import report_results


def bb(x, y):

    return -x-y, 100*(y-x*x)**2+(1-x)**2, x+y-5


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-x', type=float, required=True)
    parser.add_argument('-y', type=float, required=True)

    inputs = parser.parse_args()

    try:
        c1, f, c2 = bb(inputs.x, inputs.y)
        bb_outputs = list()
        bb_outputs.append({'name': 'bbo1', 'type': 'constraint', 'value': c1})
        bb_outputs.append({'name': 'bbo2', 'type': 'objective', 'value': f})
        if c2 < 0:
            bb_outputs.append({'name': 'bbo3', 'type': 'constraint', 'value': c2})
        else:
            bb_outputs.append({'name': 'bbo3', 'type': 'constraint', 'value': float('inf')})
        report_results(bb_outputs)     # Return objective and constraints to Orion
        print(inputs.x, inputs.y, c1, f, c2)
            
    except Exception as exception:
        raise
