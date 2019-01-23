
from EPANN import EPANN
from LunarLanderAgent import LunarLanderAgent


e = EPANN(agent_class=LunarLanderAgent, render_type='gym', N_init_hidden_nodes=0, init_IO_weights=True)

e.plotNetwork(show_plot=True, node_legend=True)






#
