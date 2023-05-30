##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## This example script created by Michael Wilson
## Department of Linguistics, Yale University
## Email: michael.a.wilson@yale.edu
## GitHub: mawilson1234
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys

import torch
from torch import nn
from copy import deepcopy
from .mixlinear import MixLinear

def replace_layer_for_mixout(module: nn.Module, mixout_prob: float) -> nn.Module:
        '''
        Replaces a single layer with the correct layer for use with Mixout.
        If module is nn.Dropout, replaces it with a Dropout where p = 0.
        If module is nn.Linear, replaces it with a MixLinear where p(mixout) = mixout_prob.
        In all other cases, returns the module unchanged.
        
            params:
                module (nn.Module)    : a module to replace for Mixout
                mixout_prob (float)   : the desired Mixout probability
            
            returns:
                module (nn.Module)    : the module set up for use with Mixout
        '''
        if isinstance(module, nn.Dropout):
            return nn.Dropout(0)
        elif isinstance(module, nn.Linear):
            target_state_dict   = deepcopy(module.state_dict())
            bias                = True if module.bias is not None else False
            new_module          = MixLinear(
                                    module.in_features,
                                    module.out_features,
                                    bias,
                                    target_state_dict['weight'],
                                    mixout_prob
                                )
            new_module.load_state_dict(target_state_dict)
            return new_module
        else:
            return module
    
def recursive_setattr(obj: 'any', attr: str, value: 'any') -> None:
    '''
    Recursively sets attributes for objects with children.
    
        params:
            obj (any)   : the object with children whose attribute is to be set
            attr (str)  : the (nested) attribute of the object, with levels indicated by '.'
                            for instance attr='attr1.attr2' sets the attr2 of obj.attr1 to
                            the passed value
            value (any) : what to set the attribute to
    '''
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)