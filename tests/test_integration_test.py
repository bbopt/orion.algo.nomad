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
from orion.algo.nomad import nomad
from orion.testing.algo import BaseAlgoTests

import itertools

class Testnomad(BaseAlgoTests):

    algo_name = "nomad"
    config = {
        "seed": 1234,  # Because this is so random
        # Add other arguments for your algorithm to pass test_configuration
        "mega_search_poll": True,
        "initial_lh_eval_n_factor": 4,
        "x0": None,
        "bb_outputs": {1: 'OBJ', 2: 'PB'}
    }


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

def test_optimizer_constraint():
    """Check functionality of Nomad Optimizer wrapper for pb with constraints."""

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(
            [
                "hunt",
                "--config",
                "./constraintRunTest/nomad.yml",
                "./constraintRunTest/bb.py",
                "-x~uniform(-5, 5, precision=None)",
                "-y~uniform(-5, 5, precision=None)",
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
                "-x~choices(['-1.2', '-0.7', '-1', '1', '-1.2', '-1.5' , '1.1', '0.3', '-0.1', '0.2', '-0.5', '0.9', '0', '0.5', '1', '1.5' , '2.0'])",
            ]
        )


def test_optimizer_choices_and_uniform():
    """Check functionality of Nomad Optimizer wrapper for shaped dimensions."""

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(
            [
                "hunt",
                "--config",
                "./benchmark/nomad.yaml",
                "./benchmark/modif_rosenbrock_2.py",
                "-x~choices(['-1.2', '-0.7', '-1', '1', '-1.2', '-1.5' , '1.1', '0.3', '-0.1', '0.2', '-0.5', '0.9', '0', '0.5', '1', '1.5' , '2.0'])",
                "-y~uniform(-1,1)"
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

def test_is_done_cardinality(self):
        """Test that algorithm will stop when cardinality is reached"""
        space = self.update_space(
            {
                "x": "uniform(0, 4, discrete=True)",
                "y": "choices(['a', 'b', 'c'])",
                "z": "uniform(1, 6, discrete=True)",
            }
        )
        space = self.create_space(space)
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(space=space)
        for i, (x, y, z) in enumerate(itertools.product(range(5), "abc", range(1, 7))):
            assert not algo.is_done
            n = algo.n_suggested
            algo.observe([[x, y, z]], [dict(objective=i)])
            assert algo.n_suggested == n + 1

        assert i + 1 == space.cardinality

        assert algo.is_done


def test_is_done_max_trials(self):
    """Test that algorithm will stop when max trials is reached"""

    algo = self.create_algo()
    self.force_observe(self.max_trials, algo)


#mySpace=Space()
# test_seeding(mySpace)
# test_optimizer_choices_and_uniform()
#test=Testnomad()
#test_is_done_max_trials(test)
#test_optimizer_constraint()
test_optimizer_choices_and_uniform()
