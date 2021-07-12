#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint:disable=invalid-name
"""Perform integration tests for `orion.algo.nomad`."""
import os

import numpy
import orion.core.cli
import pytest
from orion.algo.space import Integer, Real, Space
from orion.client import get_experiment
from orion.testing.state import OrionState
from orion.core.worker.primary_algo import PrimaryAlgo


# pylint:disable=unused-argument
def rosenbrock_function(x, y):
    """Evaluate a n-D rosenbrock function."""
    z = x - 34.56789
    r = 4 * z ** 2 + 23.4
    return [dict(name="objective", type="objective", value=r)]


# @pytest.fixture()
def space():
    """Return an optimization space"""
    space = Space()
    dim1 = Real("yolo1", "uniform", -3, 6)
    space.register(dim1)
    dim2 = Real("yolo2", "uniform", 0, 1)
    space.register(dim2)

    return space

def test_optimizer_basic():
    """Check functionality of Nomad Optimizer wrapper for single shaped dimension."""

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(
            [
                "hunt",
                "--config",
                "./benchmark/nomad.yaml",
                "./benchmark/modif_rosenbrock.py",
                "-x~uniform(-5, 5, precision=None)",
            ]
        )

def test_optimizer_choices():
    """Check functionality of Nomad Optimizer wrapper for single shaped dimension."""

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(
            [
                "hunt",
                "--config",
                "./benchmark/nomad.yaml",
                "./benchmark/modif_rosenbrock.py",
                "-x~choices(['A', 'B' , 'C', 'D', 'E', 'F'])",
            ]
        )



def test_seeding(space):
    """Verify that seeding makes sampling deterministic"""
    optimizer = PrimaryAlgo(space, "nomad")

    optimizer.seed_rng(1)
    a = optimizer.suggest(1)[0]
    with pytest.raises(AssertionError):
        numpy.testing.assert_equal(a, optimizer.suggest(1)[0])

    print(a)
    optimizer.seed_rng(1) 
    b = optimizer.suggest(1)[0]
    print(b)

    #optimizer.seed_rng(1)
    #numpy.testing.assert_equal(a, optimizer.suggest(1)[0])

#mySpace=space()
# test_seeding(mySpace)
test_optimizer_choices()

