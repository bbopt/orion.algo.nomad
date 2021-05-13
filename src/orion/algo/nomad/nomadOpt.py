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

    def __init__(self, space, seed=0):
        super(MeshAdaptiveDirectSearch, self).__init__(space, seed=seed)
       
        
        # For sampled point ids
        self.sampled = set()
 
        # Create Nomad parameters
        dimension_string = 'DIMENSION '+ str(len(self.space.values()))
        
        # Todo
        max_bb_eval_string = 'MAX_BB_EVAL 5'
        
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

        cache_file_string = 'CACHE_FILE cache.txt'

        #suggest_algo = 'MEGA_SEARCH_POLL yes' # Mads MegaSearchPoll for suggest
        suggest_algo = 'LH_EVAL 5'

        # IMPORTANT
	# Seed is managed explicitely with PyNomad.setSeed. Do not pass SEED as a parameter
        self.seed = seed       
 
        self.params = ['DISPLAY_DEGREE 2', dimension_string, max_bb_eval_string, bbo_type_string,lb_string, ub_string, cache_file_string, suggest_algo ]
        
        
        # Todo manage variable type -> bb_input_type
        for dimension in self.space.values():

#            if dimension.type != 'fidelity' and \
#                    dimension.prior_name not in ['uniform', 'int_uniform']:
#                raise ValueError("Nomad now only supports uniform and uniform discrete as prior.")

            shape = dimension.shape
            if shape and len(shape) != 1:
                raise ValueError("Nomad now only supports 1D shape.")

#            if dimension.type == 'real':
#                bb_input_type_string += 'R '
#            elif dimension.type == 'integer' and dimension.prior_name == 'int_uniform':
#                bb_input_type_string += 'I '
#            else:
#                raise NotImplementedError()
#
#            bb_input_type_string += ' )'


        # counter to deal with number of iterations: needed to properly kill the daemon thread
        self.n_iters = 0

        # list to keep candidates for an evaluation
        self.stored_candidates = list()
        
        print(self.params)
        
        
    # Not sure that it is needed
    def __del__(self):
        pass

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the Nomad random number generator.

        .. note:: This methods does nothing if the algorithm is deterministic.
        """
        self.seed = seed
        
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
        #print(self.rng_state)
        return {"rng_state": self.rng_state, "sampled": self.sampled}

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
                
        self.rng_state = state_dict["rng_state"]
        self.sampled = state_dict["sampled"]
        
        #print(self.rng_state)
        PyNomad.setSeed(self.seed)
        PyNomad.setRNGState(self.rng_state)
        
    def suggest(self, num=1):
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
       
        print(self.params)
        
        self.stored_candidates = PyNomad.suggest(self.params)        
 
        assert len(self.stored_candidates) > 0, "At least one candidate must be provided !"

        print("Suggest: ",self.stored_candidates)
        
        # Todo manage prior conversion : candidates -> samples
        samples = []
        for point in self.stored_candidates:
            point = regroup_dims(point, self.space)
            samples.append(point)
            if len(samples) >= num:   # return the number requested.
                break;

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
        
        outputs_candidates = list()
        
        # collect outputs
        for candidate in self.stored_candidates:
            idPoint = [candidate == point for point in points]
            idResult = np.argwhere(idPoint)[0].item() # pick the first index
            outputs_candidates.append(results[idResult])
        
        print("Call PyNomad observe")
        updatedParams = PyNomad.observe(self.params,output_candidates,self.stored_candidates,"cache1.txt")

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

        
        super(MeshAdaptiveDirectSearch, self).observe(points, results)

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
