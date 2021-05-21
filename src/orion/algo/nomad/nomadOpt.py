# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.nomad.nomadOpt -- TODO
=============================================

.. module:: nomadOpt
    :platform: Unix
    :synopsis: TODO

TODO: Write long description
"""
import numpy as np

#import threading
#import queue
#import multiprocessing

from orion.algo.base import BaseAlgorithm
from orion.core.utils.points import flatten_dims, regroup_dims

import os

import PyNomad


class MeshAdaptiveDirectSearch(BaseAlgorithm):
    """Nomad is a Mesh Adaptive Direct Search (MADS) algorithm for blackbox optimization.
    
    For more information about MADS
    
    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: int
        Seed of Nomad random number generator.
        Default: 0
    
    """

    requires_dist = "Linear"
    requires_shape = "flattened"

    # Flag to use LH_EVAL or MEGA_SEARCH_POLL
    use_initial_params = True

    nomad_seed = 1

    def __init__(self, space, seed=0):
        super(MeshAdaptiveDirectSearch, self).__init__(space, seed=seed)

        print("Init called")

        # For sampled point ids
        self.sampled = set()
 
        # Create Nomad parameters
        dimension_string = 'DIMENSION '+ str(len(self.space.values()))
        
        # Todo
        # max_bb_eval_string = 'MAX_BB_EVAL 5'
                

        # Todo
        lb_string = 'LOWER_BOUND ( '
        ub_string = 'UPPER_BOUND ( '
        for interval in self.space.interval():
             lb_string += str(interval[0]) + ' '
             ub_string += str(interval[1]) + ' '
        lb_string += ' )'
        ub_string += ' )'
 
        
        # Todo
        bbo_type_string = 'BB_OUTPUT_TYPE OBJ '         

        self.cache_file_name = 'cache.txt'
        if os.path.exists(self.cache_file_name):
            os.remove(self.cache_file_name)
        cache_file_string = 'CACHE_FILE '+self.cache_file_name

        suggest_algo = 'MEGA_SEARCH_POLL yes' # Mads MegaSearchPoll for suggest after first suggest
        first_suggest_algo = 'LH_EVAL ' + str(len(self.space.values())*2)

        # IMPORTANT
	    # Seed is managed explicitely with PyNomad.setSeed. Do not pass SEED as a parameter
        self.seed = seed       
 
        
        # Todo manage variable type -> bb_input_type
        bb_input_type_string = 'BB_INPUT_TYPE ( '
        for dimension in self.space.values():

            if dimension.type != 'fidelity' and \
                    dimension.prior_name not in ['uniform', 'int_uniform']:
                raise ValueError("Nomad now only supports uniform and uniform discrete as prior.")

            shape = dimension.shape
            if shape and len(shape) != 1:
                raise ValueError("Nomad now only supports 1D shape.")

            if dimension.type == 'real':
                bb_input_type_string += 'R '
            elif dimension.type == 'integer' and dimension.prior_name == 'int_uniform':
                bb_input_type_string += 'I '
            else:
                raise NotImplementedError()

            bb_input_type_string += ' )'

        self.base_params = ['DISPLAY_DEGREE 3', dimension_string, bb_input_type_string, bbo_type_string,lb_string, ub_string, cache_file_string ]
        self.initial_params = ['DISPLAY_DEGREE 3', dimension_string, bb_input_type_string, bbo_type_string,lb_string, ub_string, cache_file_string, first_suggest_algo ]
        self.params = ['DISPLAY_DEGREE 3', dimension_string, bb_input_type_string, bbo_type_string,lb_string, ub_string, cache_file_string, suggest_algo ]

        # counter to deal with number of iterations: needed to properly kill the daemon thread
        self.n_iters = 0

        # list to keep candidates for an evaluation
        self.stored_candidates = list()
        
        # print(self.params)
        
    # Not sure that it is needed
    def __del__(self):
        pass

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the Nomad random number generator.

        .. note:: This methods does nothing if the algorithm is deterministic.
        """
        self.seed = seed

        print("Seed rng : ", seed)

        PyNomad.setSeed(seed)
        self.rng_state = PyNomad.getRNGState()
          
        # The seed is passed in the Nomad parameters.
        #try:
        #   found = False
        #   for i in range(len(self.params)):
        #      split_param = self.params[i].split()
        #      if ( split_param[0].upper() == "SEED" ):
        #         self.params[i] = "SEED " + str(seed)
        #         found = True
        #         break;
        #   if not found:
        #      self.params.append("SEED " + str(seed))

	#   # Need to reset Nomad RNG to make active the change of seed
        #   # print("Reset RNG",seed)
        #   PyNomad.resetRandomNumberGenerator()
        #except AttributeError:
        #   pass 
#
    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        
        self.rng_state = PyNomad.getRNGState()
        print("State dict : ",self.rng_state)
        return {"rng_state": self.rng_state, "sampled": self.sampled}

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
                
        self.rng_state = state_dict["rng_state"]
        self.sampled = state_dict["sampled"]
        
        print("Set state : ",state_dict)
        PyNomad.setSeed(self.seed)
        PyNomad.setRNGState(self.rng_state)


    def suggest(self, num=None):
        """Suggest a `num`ber of new sets of parameters.

        TODO: document how suggest work for this algo

        Parameters
        ----------
        num: int, optional
            Number of points to suggest. Defaults to 1.

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

        print(self)

        # print('Use initial params : ' , self.use_initial_params)

        # TEMP for testing
        # MeshAdaptiveDirectSearch.nomad_seed += 1
        #PyNomad.setSeed(MeshAdaptiveDirectSearch.nomad_seed)
        #print(MeshAdaptiveDirectSearch.nomad_seed)
        #print(PyNomad.getRNGState())
        print('RNG State: ',self.rng_state)

        if self.use_initial_params:
            print("Params for suggest:",self.initial_params)
            self.stored_candidates = PyNomad.suggest(self.initial_params, self.rng_state)
            #self.first_suggest = False
            #self.first_suggestInt = 0
        else:
            print("Params for suggest:", self.params)
            self.stored_candidates = PyNomad.suggest(self.params, self.rng_state)

        # print('First suggest: ' , self.first_suggest)
        assert len(self.stored_candidates) > 0, "At least one candidate must be provided !"

        print("Suggest: ",self.stored_candidates)

        rngSeed=PyNomad.getRNGState()
        print('PyNomad RNG seed :',rngSeed)

        # Todo manage prior conversion : candidates -> samples
        samples = []
        for point in self.stored_candidates:
            point = regroup_dims(point, self.space)
            samples.append(point)
            # if len(samples) >= num:   # return the number requested.
            #    break;

        num = len(samples)
        return samples

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.
        
        Feed an observation back to Nomad callback objective function.

        TODO: document how observe work for this algo

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
        gradient : 1D array-like, optional
           Contains values of the derivatives of the `objective` function
           with respect to `params`.
        constraint : list of numeric, optional
           List of constraints expression evaluation which must be greater
           or equal to zero by the problem's definition.

        """
        assert len(points) == len(results), "The length is not the same"


        print('observe, use_first_params=',self.use_initial_params)
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
            #print(point)
            #print(flatten_dims(point,self.space))
            flat_point = flatten_dims(point,self.space)
            flat_point_tuple = list()
            for x in flat_point:
                 #print(x)
                 flat_point_tuple.append(x)
            #print(flat_point_tuple)
            candidates.append(flat_point_tuple)
     
        print("Call PyNomad observe")
        print(candidates_outputs)
        print(candidates)

        if self.use_initial_params:
             updatedParams = PyNomad.observe(self.initial_params,candidates,candidates_outputs,self.cache_file_name)
             self.use_initial_params = False  # after initial observe we use only params
        else:
            updatedParams = PyNomad.observe(self.params,candidates,candidates_outputs,self.cache_file_name)

    	# Decode bytes into string
        for i in range(len(updatedParams)):
            updatedParams[i] = updatedParams[i].decode('utf-8')
        for i in range(len(self.params)):
            if type(self.params[i]) is bytes:
                self.params[i] = self.params[i].decode('utf-8')

        print("Updated parameters by observe:\n",updatedParams)

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

        print("Parameters for next iteration:\n",self.params)
        print("\n")

        # Not sure that I need that
        # super(MeshAdaptiveDirectSearch, self).observe(points, results)

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement."""
        # NOTE: Drop if not used by algorithm
        pass

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
