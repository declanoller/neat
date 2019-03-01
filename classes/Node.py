import sys
sys.path.append('./classes')
import numpy as np
from math import exp, tanh
from copy import copy


class Node:

    def __init__(self, node_index):

        self.is_input_node = False
        self.is_output_node = False
        self.is_bias_node = False
        self.is_memory_node = False

        self.node_index = node_index

        self.input_indices = []

        self.inputs_received = []

        self.output_weights = {}

        self.value = None


    def setToInputNode(self):
        self.is_input_node = True


    def setToOutputNode(self):
        self.is_output_node = True


    def setToBiasNode(self):
        self.is_bias_node = True
        self.value = 1.0


    def setToMemoryNode(self):
        self.is_memory_node = True
        self.value = 0.0


    def getValue(self):

        if self.value is not None:
            return(self.value)
        else:
            if self.is_output_node:
                tot = sum(self.inputs_received)
                self.value = tot
                return(self.value)
            elif self.is_bias_node:
                pass
            elif self.is_memory_node:
                pass
            elif self.is_input_node:
                # For now, I'm just gonna set the input nodes directly via the .output value.
                return(self.value)
            else:
                tot = sum(self.inputs_received)
                self.value = self.nonlinear(tot)
                return(self.value)



    def calculateNodeValue(self):
        if self.is_output_node:
            tot = sum(self.inputs_received.values())
            self.value = tot
        elif self.is_bias_node:
            pass
        elif self.is_memory_node:
            pass
        elif self.is_input_node:
            # For now, I'm just gonna set the input nodes directly via the .output value.
            pass
        else:
            tot = sum(self.inputs_received.values())
            self.value = self.nonlinear(tot)


    def clearInputs(self):
        if not self.is_input_node:
            self.inputs_received = []


    def clearNode(self):
        self.clearInputs()
        self.value = None


    def setRandomOutputWeights(self):
        weights = np.random.normal(size=self.getNOutputs(), scale=0.1)
        self.output_weights = dict(zip(self.getOutputIndices(), weights))


    def removeFromInputIndices(self, ind):
        self.input_indices.remove(ind)

    def removeFromOutputWeights(self, ind):
        del self.output_weights[ind]

    def addToInputIndices(self, ind):
        self.input_indices.append(ind)


    def changeOutputWeightInd(self, old_ind, new_ind):
        weight = self.output_weights.pop(old_ind)
        self.output_weights[new_ind] = weight

    def addToOutputWeights(self, new_output_ind, val=None, std=0.1):
        if val is not None:
            self.output_weights[new_output_ind] = val
        else:
            self.output_weights[new_output_ind] = np.random.normal(scale=std)


    def mutateOutputWeight(self, ind, std=0.1):
        self.output_weights[ind] += np.random.normal(scale=std)


    def getOutputIndices(self):
        return(list(self.output_weights.keys()))


    def getNInputs(self):
        return(len(self.input_indices))

    def getNOutputs(self):
        return(len(self.output_weights))


    def getOutputWeightStr(self):
        w_str = ', '.join(['{}: {:.3f}'.format(k,v) for k,v in self.output_weights.items()])
        s = '[{}]'.format(w_str)
        return(s)

    def setOutputIndices(self, ind_list):
        self.output_weights = dict(zip(copy(ind_list), [0]*len(ind_list)))


    def setInputIndices(self, ind_list):
        self.input_indices = copy(ind_list)
        self.clearInputs()


    def allInputsReceived(self):

        #if self.input_indices is None:
        if len(self.input_indices) == 0:
            return(True)

        # checks if there are any None's left in the list. If there aren't, it has all inputs
        # and is ready to proceed.
        if list(self.inputs_received.values()).count(None)==0:
            return(True)
        else:
            return(False)




    def addToInputsReceived(self, val):
        self.inputs_received.append(val)


    def nonlinear(self, x):

        # Let's start with a nice simple sigmoid.

        #sigmoid = 1/(1 + exp(-x))
        #relu = max(0, x)
        tanh_x = tanh(x)

        return(tanh_x)


#
