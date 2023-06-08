from sympy.matrices import Matrix
from sympy import sqrt
import numpy as np

class QMC:
    def __init__(self, dim, num_states, transitions, labels, ini_state = None):
        self.dimension = dim
        self.num_states = num_states
        if self.check_transitions(transitions):
            self.transitions = transitions
        self.labels = labels
        self.ini_state = ini_state
        self.states_vec = None

    def check_transitions(self, transitions_list):
        for states_pair in transitions_list.keys():
            if states_pair[0] >=self.num_states or states_pair[1] >= self.num_states or states_pair[0] < 0 or states_pair[1] < 0:
                raise Exception('The states information is not matched, please check your input model!')
        ID_DIM = Matrix(np.identity(self.dimension))
        for ind in range(0,self.num_states):
            matrix_sum = Matrix(np.zeros((self.dimension,self.dimension)))
            for states_pair in transitions_list.keys():
                if states_pair[0] == ind:
                    for kraus_operators in transitions_list[states_pair]:
                        matrix_sum += kraus_operators.T.conjugate() * kraus_operators
            if matrix_sum != ID_DIM:
                raise Exception("The transition super-operators do not satisfy identity, please check your input model!")
        return True






