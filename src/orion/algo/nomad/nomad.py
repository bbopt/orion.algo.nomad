# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.nomad.nomad -- TODO
=============================================

.. module:: nomad
    :platform: Unix
    :synopsis: TODO

TODO: Write long description
"""
import copy

import numpy 
import os

from orion.algo.base import BaseAlgorithm
from orion.core.utils.points import flatten_dims, regroup_dims

import PyNomad


class nomad(BaseAlgorithm):
    """Nomad is a Mesh Adaptive Direct Search (MADS) algorithm for blackbox optimization.

    For more information about MADS and NOMAD: www.gerad.ca/nomad

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: int
        Seed of Nomad random number generator.
        Default: 0
    mega_search_poll: bool
        Use Mads mega search poll strategy to generate Points
        Default: True
    initial_lh_eval_n_factor: int
        Multiply the factor by n and obtain the number of latin hypercube
        samples used in the initial phase
    x0: dict
        A single initial point


    """

    requires_dist = "linear"
    requires_shape = "flattened"
    requires_type = "numerical"

    # Global flag to use LH_EVAL or MEGA_SEARCH_POLL
    use_initial_params = True

    # Global flag to stop when no points are suggestes
    no_candidates_suggested = False

    def __init__(self, space, seed=None, mega_search_poll=True, initial_lh_eval_n_factor=3, x0=None):
        super(nomad, self).__init__(space,seed=seed,
                                          mega_search_poll=mega_search_poll,
                                          initial_lh_eval_n_factor=initial_lh_eval_n_factor,
                                          x0=x0)


    @property
    def space(self):
        """Return transformed space of PyNomad"""
        return self._space

    @space.setter
    def space(self, space):
        """Set the space of PyNomad and initialize it"""
        self._original = self._space
        self._space = space
        self._initialize(space)

    def _initialize(self, space):

        assert self.mega_search_poll, "For the moment PyNomad only works with mega_search_poll activated"
        assert self.initial_lh_eval_n_factor >= 0, "PyNomad only works with initial_lh_eval_n_factor>=0"

        # For sampled point ids
        self.sampled = set()


        #
        # Create Nomad parameters
        #

        # TODO try to convert precision give to Real parameters into granularity.
        # This must consider upper and lower bounds. Not so easy!

        # Dimension, bounds and bb_input_type  for flattened space
        # X0 is not set as a Nomad parameters
        dim = 0
        dimension_string = 'DIMENSION '
        lb_string = 'LOWER_BOUND ( '
        ub_string = 'UPPER_BOUND ( '
        all_variables_are_granular = True
        bb_input_type_string = 'BB_INPUT_TYPE ( '
        lb = list()
        ub = list()
        for val in self.space.values():
            if val.type == "fidelity" :
                raise ValueError( "PyNomad does not support fidelity type" )


            if val.prior_name not in [
                "uniform",
                "reciprocal",
                "int_uniform",
                "int_reciprocal",
                "choices",
            ]:
                raise ValueError(
                    "PyNomad now only supports uniform, loguniform, uniform discrete, choices"
                    f" as prior: {val.prior_name}"
                )

            shape = val.shape

            if shape and len(shape) != 1:
                raise ValueError("PyNomad now only supports 1D shape.")
            elif len(shape) == 0:
                dim += 1
                lb.append(val.interval()[0])
                ub.append(val.interval()[1])
                lb_string += str(val.interval()[0]) + ' '
                ub_string += str(val.interval()[1]) + ' '
                if val.type == 'real':
                    bb_input_type_string += 'R '
                    all_variables_are_granular = False
                elif val.type == 'integer' :
                    bb_input_type_string += 'I '
                else :
                    raise ValueError("PyNomad now only accepts real and integer type ")
            else:
                dim += shape[0]
                for s in range(shape[0]):
                    lb.append(val.interval()[0])
                    ub.append(val.interval()[1])
                    lb_string += str(val.interval()[0]) + ' '
                    ub_string += str(val.interval()[1]) + ' '
                    if val.type == 'real':
                        bb_input_type_string += 'R '
                        # all_variables_are_granular = False
                    elif val.type == 'integer' :
                        bb_input_type_string += 'I '
                    else :
                        raise ValueError("PyNomad now only accepts real and integer type ")

        dimension_string += str(dim)
        lb_string += ' )'
        ub_string += ' )'
        bb_input_type_string += ' )'


        # Manages x0 obtained by the Algorithm configuration: dictionary {'x': 0.1, 'y': 0.4} -> must be consistent with space (dimension, input_type)
        # TODO handle multiple points
        self.x0_transformed = list()
        point = list()
        if self.x0 is not None:
            assert type(self.x0) is dict, "PyNomad: x0 must be provided as a dictionary"
            for val in self.space.values():
                point.append(self.x0[val.name[1:]])

            assert len(point) == dim, "PyNomad: x0 dimension must be consistent with variable definition"

            # Transform the x0 provided in user space into the optimization space.
            # Suggest provide points in optimization space
            for i in range(dim):
                self.x0_transformed.append(self.space.transform(point)[i].tolist())

            assert ub >= self.x0_transformed >= lb, "x0 must be within bounds"

        if not self.x0 and self.initial_lh_eval_n_factor == 0:
            raise ValueError("PyNomad needs an initial phase: provide x0 or initial_lh_eval_n_factor>0 ")

        # Todo constraints
        bbo_type_string = 'BB_OUTPUT_TYPE OBJ '

        self.cache_file_name = 'cache'+str(os.getpid())+'.txt'
        if os.path.exists(self.cache_file_name):
            os.remove(self.cache_file_name)
            self.use_initial_params = True
        cache_file_string = 'CACHE_FILE '+self.cache_file_name

        suggest_algo = 'MEGA_SEARCH_POLL yes' # Mads MegaSearchPoll for suggest after first suggest

        first_suggest_algo = 'LH_EVAL ' + str(dim*self.initial_lh_eval_n_factor)

        # IMPORTANT
        # Seed is managed explicitly with PyNomad.setSeed. Do not pass SEED as a parameter

        # if all_variables_are_granular:
        #    self.max_calls_to_extra_suggest = 100 * pow(3,len(self.space.values())) # This value is MAX_EVAL in Nomad when all variables are granular
        #else :
        self.max_calls_to_extra_suggest = 1 # This is arbitrary  # TODO make this a PyNomad parameter

        # If x0 is defined the first suggest is to provide x0 (initial params is not used)
        # + additional LH samples if required (initial params is used)
        self.initial_params = ['DISPLAY_DEGREE 2', dimension_string, bb_input_type_string, bbo_type_string,lb_string, ub_string, cache_file_string, first_suggest_algo ]
        self.params = ['DISPLAY_DEGREE 2', dimension_string, bb_input_type_string, bbo_type_string,lb_string, ub_string, cache_file_string, suggest_algo ]

        # print(self.initial_params, self.params)

        # list to keep candidates for an evaluation
        self.stored_candidates = list()


    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.

        .. note:: This methods does nothing if the algorithm is deterministic.
        """
        self.seed = seed

        PyNomad.setSeed(seed)
        self.rng_state = PyNomad.getRNGState()

        # print("Seed rng: ", seed,self.rng_state)


    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""

        self.rng_state = PyNomad.getRNGState()
        # print("State dict : ",self.rng_state)
        # return {'rng_state': self.rng_state, 'use_initial_params': self.use_initial_params, "_trials_info": copy.deepcopy(self._trials_info)}

        return {'rng_state': self.rng_state, "_trials_info": copy.deepcopy(self._trials_info)}

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """

        self.rng_state = state_dict["rng_state"]
        self._trials_info = state_dict.get("_trials_info")
        # self.use_initial_params =state_dict.get("use_initial_params")

        # print("Set state : ",state_dict)
        PyNomad.setRNGState(self.rng_state)

    def suggest(self, num=None):
        """Suggest a `num`ber of new sets of parameters.

        TODO: document how suggest work for this algo

        Parameters
        ----------
        num: int, optional
            Number of points to suggest. Defaults to None.

        Returns
        -------
        list of points or None
            A list of lists representing points suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.

        Notes
        -----
        New parameters must be compliant with the problem's domain `orion.algo.space.Space`.

        """

        # Clear candidates before a new suggestion
        self.stored_candidates.clear()


        # print('Use initial params : ' , self.use_initial_params)
        # print('Suggest RNG State: ',self.rng_state)

        if self.use_initial_params:
            if self.x0_transformed:
                self.stored_candidates.append(self.x0_transformed)
                # print("x0 is the first suggest")
            else:
                # print("Initial Params for suggest:",self.initial_params)
                self.stored_candidates = PyNomad.suggest(self.initial_params)
        else:
            # print("Params for suggest:", self.params)
            self.stored_candidates = PyNomad.suggest(self.params)

        #print("Suggest: ",self.stored_candidates)

        # extra suggest with LH to force suggest of candidates
        nb_suggest_tries = 0
        if self.initial_lh_eval_n_factor > 0:
            while len(self.stored_candidates) < num and nb_suggest_tries < self.max_calls_to_extra_suggest:
                self.stored_candidates.extend(x for x in PyNomad.suggest(self.initial_params) if x not in self.stored_candidates) # make sure to not add duplicate points
                nb_suggest_tries += 1
                #print("Extra Suggest (LH): ",self.stored_candidates)

        # assert len(self.stored_candidates) > 0, "At least one candidate must be provided !"

        # manage prior conversion : candidates -> samples
        samples = []
        for point in self.stored_candidates:
            # print(point)

            # Convert to integer if necessary and assert that value is not changed
            for i in range(len(point)):
                if self.space.values()[i].type == 'integer':
                    intVal=int(point[i])
                    assert intVal==point[i], 'PyNomad Suggest: point must be integer'
                    point[i]=intVal

            point = regroup_dims(point, self.space)
            if point not in samples:
                self.register(point)
                samples.append(point)
            if len(samples) >= num:   # return the number requested.
                break;

        num = len(samples)
        self.no_candidates_suggested = (num == 0 )

        # print("Suggest samples: ",samples)

        if samples:
           return samples

        return None

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        Feed an observation back to PyNomad.

        Observe puts points and corresponding results in Nomad cache file. Observe updates the mesh and frame size.
        The updated cache file and frame size are used by next suggest.


        Parameters
        ----------
        points : list of tuples of array-likes
           Points from a `orion.algo.space.Space`.
           Evaluated problem parameters by a consumer.
        results : list of dicts
           Contains the result of an evaluation; partial information about the
           black-box function at each point in `params`.

        Result
        ------
        objective : numeric
           Evaluation of this problem's objective function.
        constraint : list of numeric, optional
           List of constraints expression evaluation which must be greater
           or equal to zero by the problem's definition.

        """
        assert len(points) == len(results), "PyNomad observe: The length of results and points are not the same"

        #print('observe, use_first_params=',self.use_initial_params)
        candidates_outputs = list()
        candidates = list()
        for point, result in zip(points, results):

            #if not self.has_suggested(point):
            #    logger.info(
            #        "Ignoring point %s because it was not sampled by current algo.",
            #        point,
            #    )
            #    continue
            tmp_outputs = list()
            tmp_outputs.append(result['objective']) # TODO constraints

            candidates_outputs.append(tmp_outputs) # TODO constraints
            flat_point = flatten_dims(point,self.space)
            flat_point_tuple = list()
            for x in flat_point:
                 #print(type(x))
                 if type(x)==numpy.ndarray:
                    assert x.size==1, "PyNomad observe: The length of the ndarray should be one"
                    flat_point_tuple.append(numpy.float64(x))
                 else:
                    flat_point_tuple.append(x)
            candidates.append(flat_point_tuple)

        # print("Call PyNomad observe")
        # print(candidates_outputs)
        # print(candidates)

        if self.use_initial_params:
             # print("Initial params:",self.initial_params)
             updatedParams = PyNomad.observe(self.initial_params,candidates,candidates_outputs,self.cache_file_name)
             self.use_initial_params = False  # after initial observe we use only params
        else:
             updatedParams = PyNomad.observe(self.params,candidates,candidates_outputs,self.cache_file_name)


        super(nomad, self).observe(points, results) 


        # Decode bytes into string
        for i in range(len(updatedParams)):
            updatedParams[i] = updatedParams[i].decode('utf-8')
        for i in range(len(self.params)):
            if type(self.params[i]) is bytes:
                self.params[i] = self.params[i].decode('utf-8')

        # print("Updated parameters by observe:\n",updatedParams)

        # Replace updated params in params OR add if not present
        for i in range(len(updatedParams)):
            split1 = updatedParams[i].split()
            found = False
            for j in range(len(self.params)):
                split2 = self.params[j].split()
                if ( split2[0].upper() == split1[0].upper() ):
                    self.params[j] = updatedParams[i]
                    found = True
                    break;
            if not found:
                self.params.append(updatedParams[i])

        #print("Parameters for next iteration:\n",self.params)
        #print("\n")

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement."""
        # NOTE: Drop if base implementation is fine.
        return self.no_candidates_suggested or super(nomad, self).is_done

    def score(self, point):
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance.
        """
        # NOTE: Drop if not used by algorithm
        pass

    def judge(self, point, measurements):
        """Inform an algorithm about online `measurements` of a running trial."""
        # NOTE: Drop if not used by algorithm
        pass

    @property
    def should_suspend(self):
        """Allow algorithm to decide whether a particular running trial is still
        worth to complete its evaluation, based on information provided by the
        `judge` method.

        """
        # NOTE: Drop if not used by algorithm
        pass
