#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple one dimensional example for a possible user's script."""
import argparse

import numpy

from orion.client import report_results


def rosenbrock_function(x,y):
    """Evaluate a 2-D rosenbrock like function."""
    summands = 100 * (x - x**2)**2 + (1 - x)**2 + y**2
    return summands 


def execute():
    """Execute a simple pipeline as an example."""
    # 1. Receive inputs as you want
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', type=float, required=True,
                        help="Representation of a floating number")
    parser.add_argument('-y', type=float, required=True)

    inputs = parser.parse_args()

    # 2. Perform computations
    f = rosenbrock_function(inputs.x, inputs.y)

    # 3. Gather and report results
    results = list()
    results.append(dict(
        name='rosenbrock',
        type='objective',
        value=f))
    report_results(results)


if __name__ == "__main__":
    execute()
