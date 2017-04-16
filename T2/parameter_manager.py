"""
Parameter Manager
Computes all possible combinations of parameters specified in the parameter settings.

Members:
paramter_sets: 
    A list of parameter sets. A parameter set is a list of parameters used in one computation. 
    A parameter is a dictionary with keys = parameter_name, values = parameter_value).
    Computed by the method compute_parameter_sets
"""

class ParameterManager:

    def __init__(self):
        self.parameter_sets = []

    """
    Computes all possible combinations of parameters in the specified range of the parameter setting.
    
    arguments:
        parameter_settings: a list of parameter settings. A parameter setting is a touple (parameter_name, parameter_range)
        parameter_set: used for recursion. Does not need to be specified.
    """
    def compute_parameter_sets(self, parameter_settings, parameter_set=None):
        if not parameter_set:
            parameter_set = {}

        if not parameter_settings:
            self.parameter_sets.append(parameter_set)

        else:
            ps = parameter_settings.pop()

            for i in ps[1]:
                temp = dict(parameter_set)
                temp[ps[0]] = i
                self.compute_parameter_sets(parameter_set=temp, parameter_settings=list(parameter_settings))

settings = [('alpha', range(1,5)),('beta', range(1,15))]
manager = ParameterManager()
manager.compute_parameter_sets(parameter_settings=settings)
print [x for x in manager.parameter_sets]





