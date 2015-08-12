
import random #as random
import pylab as plt
import scipy as SP
import networkx as nx
import math
import os
from bitstring import BitArray, BitStream

#  CHECK IF DIRRECTORIES IN FILE PATH EXIST. IF NOT, CREATE THEM.
def ensure_dirs_exist(f):
	d = os.path.dirname(f)
	if not os.path.exists(d):
		os.makedirs(d)

# INITIAILIZE VARIABLES
def init():
	global time, time_steps, depth, find, forget, network, positions
	time 			= 0
	time_steps 		= 10
	depth			= 2		# Depth of tree traversed to make a connection	(could be a list to miss those close)
	find			= 0.25 	# Chance of gaining each of a connections strength from a link
	forget			= 0.15 	# Chance of forgetting each skill in each time step. Later more for new ones?
	
	nodes 			= 10	# The number of nodes
	k 				= 4		# Each node is connected to k nearest neighbors in ring topology
	p 				= 0.03	# The probability of rewiring each edge
	seed			= None	# Seed for random number generator (default=None)
	len_resources 	= 5
	network = nx.watts_strogatz_graph(nodes, k, p, seed)
	for n in network.nodes_iter():
		r = random.randint(0, 2**len_resources - 1)
		network.node[n]['resources'] = BitArray(uint=r, length=len_resources)
		network.node[n]['score'] 	 = 0
		for r in range(len(network.node[n]['resources'])):
			if network.node[n]['resources'][r] == True:
				network.node[n]['score'] += 1			
	positions = nx.spring_layout(network)
	nextNetwork = network.copy()

# PLOT OUTPUT AND AND SAVE FIGURE
def draw():
		plt.cla()
		res = [100 * network.node[i]['score'] for i in network.nodes_iter()]
		nx.draw_networkx(network,positions,edgelist=network.edges(),
			node_size=res,with_labels=False)
		plt.axis('image')
		plt.title('Network Plot')
		filename = 'Plots/Time_step_' + str(time) + '.png'
		ensure_dirs_exist(filename)	
		plt.savefig(filename)	

# LOGIG FOR EACH TIME-STEP
def step():
	for n in network.nodes_iter():
		nbrs = network.neighbors(n)
		# Find: For each '1' in neighbours 'resources' and gain a '1' with probability 'find'
		for nbr in nbrs:
			for r in range(len(network.node[nbr]['resources'])):
				if network.node[nbr]['resources'][r] == True:
					if random.uniform(0, 1) < find:
						if (network.node[n]['resources'][r] | network.node[nbr]['resources'][r]):
							network.node[n]['resources'][r] = True
		# Forget: switch each '1' in 'resources' to a '0' with probablity 'forget'
		for r in range(len(network.node[n]['resources'])):
			if network.node[n]['resources'][r] == True:
				if random.uniform(0, 1) < forget:
					network.node[n]['resources'][r] = False
		# Score: count the number of '1s' in 'resources'
		network.node[n]['score'] = 0
		for r in range(len(network.node[n]['resources'])):
			if network.node[n]['resources'][r] == True:
				network.node[n]['score'] += 1

# MAIN LOOP
if __name__ == '__main__':
	init()
	draw()
	
	for time in range(time_steps):
		step()
		draw()


