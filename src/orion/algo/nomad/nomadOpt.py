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

from orion.algo.base import BaseAlgorithm
from orion.core.utils.points import flatten_dims, regroup_dims

from PyNomad import optimize as nomad_solve

class PyNomadOptimizer(BaseAlgorithm):
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

    def __init__(self, space, seed=0):
        super(PyNomadOptimizer, self).__init__(space, seed=seed)
        
        # Create Nomad parameters
        # bb_input_type_string = 'BB_INPUT_TYPE ( '
        dimension_string = 'DIMENSION '+ len(self.space.values())
        
        # Todo
        max_bb_eval_string = 'MAX_BB_EVAL 10'
        
        # lb_string = 'LOWER_BOUND ( '
        # ub_string = 'UPPER_BOUND ( '
        
        params = [dimension_string,
                  max_bb_eval_string]
        
        lb = [] # Todo
        ub = [] # Todo
        x0 = [] # Todo or use LH Sampling with n_initial_points
        
        
        # Todo manage variable type -> bb_input_type
        for dimension in self.space.values():

            if dimension.type != 'fidelity' and \
                    dimension.prior_name not in ['uniform', 'int_uniform']:
                raise ValueError("Nomad now only supports uniform and uniform discrete as prior.")

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


        # queues to communicate between threads
        self.inputs_queue = multiprocessing.JoinableQueue()
        self.outputs_queue = multiprocessing.JoinableQueue()
        
        # counter to deal with number of iterations: needed to properly kill the daemon thread
        self.n_iters = 0

        # list to keep candidates for an evaluation
        self.stored_candidates = list()
        
        # start background thread
        self.nomad_process = multiprocessing.Process(target=nomad_solve, args=(self.bb_fct, x0, lb, ub, params,))
        self.nomad_process.start()
        
        
    # Not sure that it is needed
    def __del__(self):
        self.nomad_process.terminate()
        self.nomad_process.join()

    # Nomad solve callback objective function
    def bb_fct(self, x):
        try:
            n_values = x.get_n()

            dim_pb = len(self.space.values())
            
            if ( n_values != dim_pb ):
                print("Invalid number of values passed to bb")
                return -1

            # store the input points
            candidates = []
            for i in range(n_pts):
                candidates.append([x.get_coord(j) for j in range(i*dim_pb,(i+1)*dim_pb)])
            
            #  print("candidates")
            #  for candidate in candidates:
            #      print(candidate)
            
            self.inputs_queue.put(candidates)
            self.inputs_queue.join()

            # wait until the blackbox returns observations
            while self.outputs_queue.empty():
                continue
                
            # Todo test the size of bb output

            # returns observations to the blackbox
            outputs_candidates = self.outputs_queue.get()
            for output_val in outputs_candidates:
                x.set_bb_output(i, output_val)

            # task finish
            self.outputs_queue.task_done()

        except:
            print ("Unexpected error in bb()", sys.exc_info()[0])
            return -1
        return 1



    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the Nomad random number generator.

        .. note:: This methods does nothing if the algorithm is deterministic.
        """
        self.rng = seed

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        # TODO: Adapt this to your algo
        return {'rng_state': self.rng.get_state()}

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        # TODO: Adapt this to your algo
        self.seed_rng(0)
        self.rng.set_state(state_dict['rng_state'])

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
        
        if num > 1:
            raise ValueError("Nomad should suggest only one point.")
          
        # Clear candidates before a new suggestion
        self.stored_candidates.clear()
        
        # Wait until Nomad gives candidates
        while self.inputs_queue.empty():
            continue
        
        # collect candidates from objective callback function
        candidates = self.inputs_queue.get()
            
        assert len(candidates) == 1, "Only one candidate must be provided !"

        # Todo manage prior conversion : candidates -> samples
        
        for point in candidates:
            point = regroup_dims(point, self.space)
            samples.append(point)
            
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
        
        # trigger callbacks
        if self.nomad_process.is_alive():
        #  if self.nomad_thread.is_alive():
            self.outputs_queue.put(outputs_candidates)
            # wait for completion
            self.outputs_queue.join()
            print("Observation passed back to Nomad!")
        
        super(TPE, self).observe(points, results)

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
