import numpy as np
from queue import Queue
from Node import Node
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import random
import networkx as nx
import pygraphviz as pgv
#from torch.distributions import Categorical
#import torch

class EPANN:

    def __init__(self, agent_class, **kwargs):

        self.agent = agent_class(**kwargs)

        self.render_type = kwargs.get('render_type', 'matplotlib')
        self.verbose = kwargs.get('verbose', False)
        self.action_space_type = kwargs.get('action_space_type', 'discrete')
        self.N_init_hidden_nodes = kwargs.get('N_init_hidden_nodes', 0)
        self.init_IO_weights = kwargs.get('init_IO_weights', False)
        self.N_inputs = self.agent.N_state_terms
        self.N_action_outputs = self.agent.N_actions
        self.N_total_outputs = self.N_action_outputs # + 3
        self.epsilon = 0.0

        self.input_node_indices = []
        self.output_node_indices = []

        self.node_list = []
        self.weights_list = set() # This will be a set of tuples of the form (parent index, child index)
        self.propagate_order = []

        self.weight_change_chance = 0.98
        self.weight_add_chance = 0.09
        self.weight_remove_chance = 0.05
        self.node_add_chance = 0.0005



        # Add bias node
        self.addNode(is_bias_node=True)

        # Add input and output nodes
        [self.addNode(is_input_node=True) for i in range(self.N_inputs)]
        [self.addNode(is_output_node=True) for i in range(self.N_total_outputs)]
        #print('input indices:', self.input_node_indices)
        #print('output indices:', self.output_node_indices)
        #self.plotNetwork()

        # Add connections from all inputs to all outputs (maybe disable later?)
        if self.init_IO_weights:
            [[self.addConnectingWeight((i, o), std=1.0) for o in self.output_node_indices] for i in self.input_node_indices]

        self.sortPropagateOrder()




    def addNode(self, is_input_node=False, is_output_node=False, is_bias_node=False):

        new_node = Node(len(self.node_list))

        if is_input_node:
            new_node.setToInputNode()
            self.input_node_indices.append(new_node.node_index)

        if is_output_node:
            new_node.setToOutputNode()
            self.output_node_indices.append(new_node.node_index)

        if is_bias_node:
            new_node.setToBiasNode()
            new_node.value = 1
            self.bias_node_index = new_node.node_index

        self.node_list.append(new_node)
        self.sortPropagateOrder()
        return(new_node.node_index)


    def addConnectingWeight(self, weight_parchild_tuple, val=None, std=0.1):

        # You shouldn't be calling this unless you already know it doesn't have that connection.
        assert weight_parchild_tuple not in self.weights_list, 'Problem in addConnectingWeight!'

        parent_node_index, child_node_index = weight_parchild_tuple
        self.node_list[child_node_index].addToInputIndices(parent_node_index)
        self.node_list[parent_node_index].addToOutputWeights(child_node_index, val=val, std=std)
        self.weights_list.add(weight_parchild_tuple)
        self.sortPropagateOrder()


    def removeConnectingWeight(self, weight_parchild_tuple):

        # You shouldn't be calling this unless you already know it has that connection.
        assert weight_parchild_tuple in self.weights_list, 'Problem in removeConnectingWeight!'

        parent_node_index, child_node_index = weight_parchild_tuple
        self.node_list[child_node_index].removeFromInputIndices(parent_node_index)
        self.node_list[parent_node_index].removeFromOutputWeights(child_node_index)
        self.weights_list.remove(weight_parchild_tuple)
        self.sortPropagateOrder()


    def addNodeInBetween(self, par_index, child_index):
        # This adds a node in between existing nodes par_index and child_index, where the output of par_index went to child_index.
        # Pass it the index of each.
        self.print('adding node between nodes {} and {}'.format(par_index, child_index))
        # Add node:
        # New node:
        old_weight = self.node_list[par_index].output_weights[child_index]
        self.removeConnectingWeight((par_index, child_index))
        new_node_index = self.addNode()
        self.addConnectingWeight((par_index, new_node_index), val=old_weight)
        self.addConnectingWeight((new_node_index, child_index), val=1)

        self.sortPropagateOrder()


    def sortPropagateOrder(self):


        # So something to keep in mind here is that if a node is isolated because its
        # node was removed or something, it will just never enter the sort_prop list.
        # That's probably fine, because it can't add anything anyway.
        #

        sort_queue = Queue()
        prop_order_set = set()
        queue_set = set()
        self.propagate_order = []

        # First add the output indices, which we'll work backwards from.
        for ind in self.output_node_indices:
            sort_queue.put(ind)
            queue_set.add(ind)

        while not sort_queue.empty():

            ind = sort_queue.get()
            queue_set.remove(ind)
            # Make sure all the children of this node are already in the list/set. If
            # one isn't, add this child to the queue if it's not already there,
            # (this is in case there's a "dead end" node that would never get seen by
            # tracing back from the outputs), put the node back in the queue, and continue
            # to the next loop iteration.
            #
            # You could also have it not break immediately, and add all its unseen children,
            # which might speed it up, but might also not.
            all_children_in_prop_order = True

            for child_ind in self.node_list[ind].getOutputIndices():
                if child_ind not in prop_order_set:
                    all_children_in_prop_order = False
                    # Only want to add the child to the queue if it isn't already in it
                    if child_ind not in queue_set:
                        sort_queue.put(child_ind)
                        queue_set.add(child_ind)

                    break

            if all_children_in_prop_order:
                # This means that the node can now be added to the prop_order and set,
                # and also add its parents to the queue if they're not already.
                self.propagate_order.append(ind)
                prop_order_set.add(ind)

                for parent_ind in self.node_list[ind].input_indices:
                    if parent_ind not in queue_set:
                        sort_queue.put(parent_ind)
                        queue_set.add(parent_ind)

            else:
                # If the children aren't all there already, put it back in the queue.
                sort_queue.put(ind)
                queue_set.add(ind)

        # Now it should be in order, where you can evaluate each node, starting with the input ones,
        # and all the inputs should arrive in the right order.
        self.propagate_order.reverse()



    def mutateAddNode(self):
        if len(self.weights_list)>0:
            par_index, child_index = random.choice(list(self.weights_list))
            self.addNodeInBetween(par_index, child_index)


    def mutateAddWeight(self, std=0.1):
        while True:
            node_1_ind = random.choice(list(range(len(self.node_list))))

            # No self
            node_2_options = [ind for ind in range(len(self.node_list)) if ind != node_1_ind]

            if (node_1_ind in self.input_node_indices) or (node_1_ind == self.bias_node_index):
                node_2_options = [ind for ind in node_2_options if (ind not in self.input_node_indices) and (ind != self.bias_node_index)]
                weight_connection_options = [(node_1_ind, ind) for ind in node_2_options if ((node_1_ind, ind) not in self.weights_list)]

            elif node_1_ind in self.output_node_indices:
                node_2_options = [ind for ind in node_2_options if ind not in self.output_node_indices]
                weight_connection_options = [(ind, node_1_ind) for ind in node_2_options if ((ind, node_1_ind) not in self.weights_list)]

            else:
                #if it's neither an input or output

                # The options if node 2 is going to be the parent.
                node_2_weight_options_parent = [(ind, node_1_ind) for ind in node_2_options if (ind not in self.output_node_indices) and (not self.getsInputFrom(ind, node_1_ind))]

                # In both cases, we need to check that either node_2 is not in prop_order
                # (meaning it can go anywhere, provided it's not i/o), OR that
                # it doesn't get indirect input from ind.
                #
                # The options if node 2 is going to be the child.
                node_2_weight_options_child = [(node_1_ind, ind) for ind in node_2_options if ((ind not in self.input_node_indices) and (ind != self.bias_node_index)) and (not self.getsInputFrom(node_1_ind, ind))]

                # Combine them.
                weight_connection_options = node_2_weight_options_parent + node_2_weight_options_child
                weight_connection_options = [w for w in weight_connection_options if w not in self.weights_list]

            if len(weight_connection_options)==0:
                # If there aren't any options by this point, continue to try again
                continue
            else:
                weight_connection_tuple = random.choice(weight_connection_options)
                break

        self.addConnectingWeight(weight_connection_tuple, val=None, std=std)


    def mutateChangeWeight(self, std=0.1):
        if len(self.weights_list)>0:
            par_index, child_index = random.choice(list(self.weights_list))
            self.print('changing weight between {} and {}'.format(par_index, child_index))
            self.node_list[par_index].mutateOutputWeight(child_index, std=std)

    def mutateRemoveWeight(self):
        if len(self.weights_list)>0:
            par_index, child_index = random.choice(list(self.weights_list))
            self.print('removing weight between {} and {}'.format(par_index, child_index))
            self.removeConnectingWeight((par_index, child_index))


    def mutate(self, std=0.1):

        self.print('\n\nbefore mutate:')
        if self.verbose:
            self.printNetwork()

        if random.random() < self.node_add_chance:
            # Add a node by splitting an existing weight
            self.mutateAddNode()


        if random.random() < self.weight_add_chance:
            # Add weight between two nodes
            self.mutateAddWeight(std=std)


        if random.random() < self.weight_change_chance:
            # Change weight
            self.mutateChangeWeight(std=std)


        if random.random() < self.weight_remove_chance:
            # Remove weight
            self.mutateRemoveWeight()


        self.print('\nafter mutate:')
        if self.verbose:
            self.printNetwork()




    def propagateNodeOutput(self, node_index):

        # This assumes that the propagate_order list is already sorted!
        # If it isn't, you'll get some bad results.
        node = self.node_list[node_index]

        for target_node_index in node.getOutputIndices():
            self.node_list[target_node_index].addToInputsReceived(node.getValue()*node.output_weights[target_node_index])


    def forwardPass(self, input_vec):

        self.clearAllNodes()

        # Put the input vec into the input nodes
        for i, index in enumerate(self.input_node_indices):
            self.node_list[index].value = input_vec[i]

        # For each node in the sorted propagate list, propagate to its children
        for ind in self.propagate_order:
            self.propagateNodeOutput(ind)

        output_vec = np.array([self.node_list[ind].getValue() for ind in self.output_node_indices])

        if self.action_space_type == 'discrete':
            action = self.epsGreedyOutput(output_vec)
        elif self.action_space_type == 'continuous':
            # Need to fix if there are several cont. directions, but won't deal with that
            # for now. Actually, it seems like even when it's one continuous action, you're
            # supposed to supply it a list??
            action = output_vec

        return(action)


    def epsGreedyOutput(self, vec):
        if random.random() < self.epsilon:
            return(random.randint(0, len(vec)-1))
        else:
            return(self.greedyOutput(vec))


    def greedyOutput(self, vec):
        return(np.argmax(vec))


    def softmaxOutput(self, vec):
        a = np.array(vec)
        a = np.exp(a)
        a = a/sum(a)
        return(np.random.choice(list(range(len(a))), p=a))





    def clearAllNodes(self):
        [n.clearNode() for i,n in enumerate(self.node_list) if i!=self.bias_node_index]


    def runEpisode(self, N_steps, plot_run=False, **kwargs):


        if plot_run:
            self.createFig()

        R_tot = 0
        Rs = []

        record_episode = kwargs.get('record_episode', False)

        if record_episode:
            self.agent.setMonitorOn()

        self.agent.initEpisode()

        for i in range(N_steps):
            self.clearAllNodes()

            if i%int(N_steps/10)==0:
                self.print('R_tot = {:.3f}'.format(R_tot))


            s = self.agent.getStateVec()
            a = self.forwardPass(s)
            self.print('s = {}, a = {}'.format(s, a))

            r, s, done = self.agent.iterate(a)

            R_tot += r
            Rs.append(R_tot)

            if done:
                return(R_tot)

            if plot_run:
                if self.render_type == 'matplotlib':
                    self.agent.drawState(self.axes[0])
                    self.axes[1].clear()
                    self.axes[1].plot(Rs)
                    self.fig.canvas.draw()
                elif self.render_type == 'gym':
                    self.agent.drawState()


        self.print('R_tot/N_steps = {:.3f}'.format(R_tot/N_steps))

        return(R_tot)




    def gaussMutate(self, std=0.1):
        # This mutates ALL of a node's output weights!
        for n in self.node_list:
            for w in n.getOutputIndices():
                n.output_weights[w] += np.random.normal(scale=std)



    def getsInputFrom(self, n1_index, n2_index):

        # This is to check if n1 gets input from n2, indirectly.

        n1 = self.node_list[n1_index]
        n2 = self.node_list[n2_index]
        lineage_q = Queue()
        [lineage_q.put(n) for n in n1.input_indices]

        while lineage_q.qsize() > 0:
            next = lineage_q.get()
            if n2_index in self.node_list[next].input_indices:
                return(True)
            else:
                [lineage_q.put(n) for n in self.node_list[next].input_indices]

        return(False)








    def clone(self):
        clone = deepcopy(self)
        return(clone)


    def createFig(self):
        if self.render_type == 'matplotlib':
            self.fig, self.axes = plt.subplots(1,2, figsize=(16,8))
            plt.show(block=False)




    def print(self, str):

        if self.verbose:
            print(str)



    def printNetwork(self):
        print('\n')
        for i, n in enumerate(self.node_list):
            print('\nnode ', i)
            print('input indices:', n.input_indices)
            print('output indices: ', n.getOutputIndices())
            print('output weights: ', n.getOutputWeightStr())

        print()


    def plotNetwork(self, show_plot=True, save_plot=False, fname=None, node_legend=False):

        fig, ax = plt.subplots(1, 1, figsize=(12,8))
        DG = nx.DiGraph()

        other_node_indices = [i for i,n in enumerate(self.node_list) if ((i not in self.input_node_indices) and (i not in self.output_node_indices) and (i != self.bias_node_index))]

        for i in self.input_node_indices:
            DG.add_node(i)

        for i in self.output_node_indices:
            DG.add_node(i)

        DG.add_node(self.bias_node_index)

        for n in self.node_list:
            for o in n.getOutputIndices():
                DG.add_edges_from([(n.node_index, o)])

        #nx.draw(DG, with_labels=True, font_weight='bold', arrowsize=20)
        pos = nx.drawing.nx_agraph.graphviz_layout(DG, prog='dot')
        nx.draw_networkx_nodes(DG, nodelist=self.input_node_indices, pos=pos, node_color='mediumseagreen', node_size=600)
        nx.draw_networkx_nodes(DG, nodelist=self.output_node_indices, pos=pos, node_color='orange', node_size=600)
        nx.draw_networkx_nodes(DG, nodelist=[self.bias_node_index], pos=pos, node_color='forestgreen', node_size=600)
        nx.draw_networkx_nodes(DG, nodelist=other_node_indices, pos=pos, node_color='orange', node_size=600)

        for w in self.weights_list:
            weight = self.node_list[w[0]].output_weights[w[1]]
            if weight < 0:
                nx.draw_networkx_edges(DG, pos=pos, edgelist=[w], width=4.0, alpha=min(abs(weight), 1), edge_color='tomato')

            if weight >= 0:
                nx.draw_networkx_edges(DG, pos=pos, edgelist=[w], width=4.0, alpha=min(abs(weight), 1), edge_color='dodgerblue')

        labels = {i:str(i) for i in range(len(self.node_list))}
        nx.draw_networkx_labels(DG, pos=pos, labels=labels, font_size=14)
        edge_labels = {w:'{:.2f}'.format(self.node_list[w[0]].output_weights[w[1]]) for w in self.weights_list}
        nx.draw_networkx_edge_labels(DG, pos=pos, edge_labels=edge_labels, font_size=10, bbox={'alpha':0.2, 'pad':0.0}, label_pos=0.85)

        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(left=.2, bottom=0, right=1, top=1, wspace=1, hspace=0)
        ax.axis('off')

        if node_legend:
            if (self.agent.state_labels is not None) and (self.agent.action_labels is not None):
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

                percent_offset = 0.02

                bias_str = 'Bias: node {}\n\n'.format(self.bias_node_index)
                input_str = bias_str + 'Inputs:\n\n' + '\n'.join(['node {} = {}'.format(ind, self.agent.state_labels[i]) for i, ind in enumerate(self.input_node_indices)])
                ax.text(-percent_offset, (1-3*percent_offset), input_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)

                output_str = 'Outputs:\n\n' + '\n'.join(['node {} = {}'.format(ind, self.agent.action_labels[i]) for i, ind in enumerate(self.output_node_indices)])
                ax.text(-percent_offset, 3*percent_offset, output_str, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=props)
                textstr = input_str + '\n\n' + output_str


                # place a text box in upper left in axes coords

        if save_plot:
            if fname is not None:
                plt.savefig(fname)
            else:
                plt.savefig(fst.getDateString() + '_NNplot.png')

        if show_plot:
            plt.show()

        plt.close()
















'''
SCRAP






        # When finished, so all output nodes should be full
        self.print('\n\nProp. done, output node values:')
        for ind in self.output_node_indices:
            self.print('Node {} output: {:.3f}'.format(ind, self.node_list[ind].value))




        if par_index != self.bias_node_index:
            self.node_list[self.bias_node_index].addToOutputWeights(new_node.node_index)
            self.node_list[self.bias_node_index].output_weights[new_node.node_index] = 0
            new_node.addToInputIndices(self.bias_node_index)



        # Bias node
        bias_node = Node(len(self.node_list))
        bias_node.setToBiasNode()
        #bias_node.setOutputIndices(self.output_node_indices)
        self.node_list.append(bias_node)


        # Add input nodes
        for i in range(self.N_inputs):
            new_node = Node(len(self.node_list))
            new_node.setOutputIndices(self.output_node_indices)
            #uself.weights_list.append()
            new_node.setRandomOutputWeights()
            new_node.setToInputNode()
            self.node_list.append(new_node)


        # Add output nodes
        for i in range(self.N_total_outputs):
            new_node = Node(len(self.node_list))
            new_node.setInputIndices(self.input_node_indices)
            #new_node.addToInputIndices(self.bias_node_index)
            new_node.setToOutputNode()
            self.node_list.append(new_node)

        # Add hidden layer nodes
        for i in range(self.N_init_hidden_nodes):
            new_node = Node(len(self.node_list))
            new_node.setInputIndices(self.input_node_indices)
            #new_node.addToInputIndices(self.bias_node_index)
            new_node.setOutputIndices(self.output_node_indices)
            new_node.setRandomOutputWeights()

            #self.node_list[self.bias_node_index].addToOutputWeights(new_node.node_index)

            for ii in self.input_node_indices:
                self.node_list[ii].addToOutputWeights(new_node.node_index)

            for o in self.output_node_indices:
                self.node_list[o].addToInputIndices(new_node.node_index)

            self.node_list.append(new_node)

        # Set initial random output weight
        for i, n in enumerate(self.node_list):
            N_incoming_connect = n.getNInputs()
            for j in n.input_indices:
                self.node_list[j].output_weights[i] = np.random.normal(scale=(1.0/N_incoming_connect))

        # Set all the bias weights to 0 to start.
        for i in self.node_list[self.bias_node_index].getOutputIndices():
            self.node_list[self.bias_node_index].output_weights[i] = 0

        self.node_list[self.bias_node_index].value = 1



if (node_1_ind not in self.propagate_order) and not ():
    # This one is easy: if it's not in propagate_order, then it's not connected to anything else,
    # so we can attach it to any other.
    node_2_ind = random.choice(node_2_options)
    if (node_2_ind in self.input_node_indices) or node_2_ind == self.bias_node_index:
        weight_connection_tuple = (node_2_ind, node_1_ind)

    elif node_2_ind in self.output_node_indices:
        weight_connection_tuple = (node_1_ind, node_2_ind)

    else:
        if random.random() < 0.5:
            weight_connection_tuple = (node_1_ind, node_2_ind)
        else:
            weight_connection_tuple = (node_2_ind, node_1_ind)

    break













'''


#
