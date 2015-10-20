import networkx as nx
import pandas as pd
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt 
import random
import math
import os
from bitstring import BitArray, BitStream
import random
from collections import OrderedDict
import copy
from itertools import islice # Is this used?

# USEFUL RESOURCES
# Docstrings format http://stackoverflow.com/questions/3898572/what-is-the-standard-python-docstring-format
# Google's Python style guide https://google-styleguide.googlecode.com/svn/trunk/pyguide.htmlfr
# Introduce ittertools http://jmduke.com/posts/a-gentle-introduction-to-itertools/
# Find substrings http://stackoverflow.com/questions/10106901/elegant-find-sub-list-in-list
# Elegant depth first search implementation http://codereview.stackexchange.com/questions/78577/depth-first-search-in-python
# Random sampling by shuffling a list then selecting http://leancrew.com/all-this/2010/08/random-sampling-with-python/
# Select pairs of nodes randomly http://stackoverflow.com/questions/10929269/how-to-select-two-nodes-pairs-of-nodes-randomly-from-a-graph-that-are-not-conn
# Maybe useful. Networkx generate a dictionary of labels from attributes. http://stackoverflow.com/questions/3982819/networkx-node-attribute-drawing
# Maybe useful. Networkx plot labels outside nodes http://stackoverflow.com/questions/11946005/label-nodes-outside-with-minimum-overlap-with-other-nodes-edges-in-networkx?rq=1
# Maybe useful. Networkx plot node color based on value http://stackoverflow.com/questions/11946005/label-nodes-outside-with-minimum-overlap-with-other-nodes-edges-in-networkx?rq=1


def ensure_dirs_exist(f):
    """Check if dirrectories in file path exist. If not create them.

    Args:
        f (str): file path.

    """
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def subfinder(resource_list, project_pattern):
    """Search for a sublist within a list.

    Args:
        resource_list (list): the list to check.
        project_pattern (list): the sublist to look for.
    Returns:
        a list of matches or an empty list if no match found.

    """    
    matches = []
    for i in range(len(resource_list)):
        if resource_list[i] == project_pattern[0] and resource_list[i:i+len(project_pattern)] == project_pattern:
            matches.append(project_pattern)
    return matches

def subfilter(resource_list, project_pattern):
    """Remove values matching a sublist from a list if the sublist is found in the list.

    Args:
        resource_list (list): the list to filter sublist values from.
        project_pattern (list): the sublist to look for.
    Returns:
        a list with characters from the sublist removed if the sublist appears in the larger list.

    """
    project_pattern = set(project_pattern)
    return [x for x in resource_list if x not in project_pattern]

def dfs(start, graph, new_graph, visit_prob):
    """Depth first search that only probabalistically visits each neighbour.

    Args:
        start (int): the index of the graph node to start with.
        graph (networkx graph): the graph to traverse. 
        visit_prob: the probability of visiting each 

    NOTE: All changes are made to a 'new_graph' which may different from 'graph' to allow this method to be called multiple times between time steps.
    TODO: Can we visit the neighbour set in random order? Randomize the set of neighbours before appending to stack list.
    TODO: If slow, consider not sorting and removing duplicats from 'resources' list in each loop.
    TODO: Add error handling to ensure that the nodes of graph and new_graph are identical. Node values and attributes may change.
    TODO: Check use of graph and new_graph is right.

    """
    visited, stack = set(), [start]
    while stack:
        # Pop a node index from the stack
        vertex = stack.pop()
        # Visit the node if 1. unvisited and 2. a random number is greater than some threshold
        if vertex not in visited:
            visited.add(vertex)
            if visit_prob > random.random():
                # Add all the 'vertex' node's neighbours to the stack to consider visiting
                stack.extend(set(graph[vertex].keys()) - visited)
                # Copy resources from the 'vertex' node to the 'start' node
                new_graph.node[start]['resources'] = new_graph.node[start]['resources'] + graph.node[vertex]['resources']
                # Remove duplicates from 'start' node's resources list and sort list
                new_graph.node[start]['resources'] = sorted(list(set(new_graph.node[start]['resources']))) 
    return new_graph

def init():
    """Initialization code runs at start of simulation.

    """
    global time, time_steps, depth, find, forget, network, new_network, positions, prob_visit, project_values_df
    time                = 0
    time_steps          = 15

    nodes               = 30    # The number of nodes
    k                   = 4     # Each node is connected to k nearest neighbors in ring topology
    p                   = 0.03  # The probability of rewiring each edge
    tries               = 100   # Number of attempts to generate a connected graph
    seed                = None  # Seed for random number generator (default=None)

    prob_visit          = 0.10  # Probability of visiting each node

    # INITIALIZE RESOURCE VALUES AND CODES
    # TODO: Should not calculate no_resources in advance but measure it so the numbe rof resoruces cannot differ from the supplied list
    no_resources    = 1*2 + 2*3 + 3*4 + 4*5 + 5*6
    resources_list  = []
    # Add a unique integer to the list to represent each resource
    for r in range(no_resources):
        resources_list.append(r)
    resources_list_temp = copy.deepcopy(resources_list)
    # Generate project values and codes and store in a Pandas Dataframe
    reward_dist = OrderedDict([ (2, [2,3,4,5,6]), (4, [3,4,5,6]), (8, [4,5,6]), (16, [5,6]), (32, [6]) ])
    project_values_df = pd.DataFrame(columns=('project_val', 'project_code'))
    res_len_check = 0
    df_loc = 0
    for key, val in reward_dist.items():
        project_val = key
        for project_length in range(len(val)):
            project_code = resources_list_temp[:val[project_length]]
            del resources_list_temp[:val[project_length]]
            res_len_check += val[project_length]
            project_values_df.loc[df_loc] = [project_val, project_code] #[random.randint(-1,1) for n in range(2)]
            df_loc +=1
    project_values_df.loc[df_loc] = [project_val, [1]]            

    # INITIALIZE NETWORK AND NODE ATTRIBUTES (RESOURCES AND JOBS/SCORE)
    network = nx.connected_watts_strogatz_graph(nodes, k, p, tries, seed)
    for n in network.nodes_iter():
        network.node[n]['resources'] = []
        network.node[n]['jobs']      = 0

    # ASSIGN RESOURCES TO NETWORK
    # Shuffle resources then just choose in order
    random.shuffle(resources_list)
    # Assign each resource chosen to one node randomly selected with replacement from the network
    for r in range(no_resources):
        n = random.choice(network.nodes())
        network.node[n]['resources'].append(resources_list[0])
        del resources_list[0] # Delete resources after assigning, so list at end is '0'. Another reason? 
        # TODO: Pop off of the stack instead of selecting and deleting to simplify.
        # TODO: Check list has length 0 at end of loop.
        # TODO: Check distribution of resources is correct before and after sorting
    # Sort resources at each node
    for n in network.nodes_iter():
        network.node[n]['resources'] = sorted(network.node[n]['resources'])

    # SET GRAPH POSITIONS AND MAKE A COPY OF THE NETWORK FOR THE NEXT TIMESTEP
    positions = nx.spring_layout(network)
    new_network = network.copy()

def draw():
    """Plotting code runs at start of simulation and after each time step.

    """
    plt.cla()

    # For plotting, try generating a dictinary with labels for each node
    labels_list=dict((n,d['resources']) for n,d in network.nodes(data=True))
    # print labels_list

    res = [100 * network.node[i]['jobs'] for i in network.nodes_iter()]
    nx.draw_networkx(network,positions,edgelist=network.edges(),
        node_size=res,with_labels=True, labels=labels_list)
    plt.axis('image')
    plt.title('Network Plot')
    filename = 'Plots/Time_step_' + str(time) + '.png'
    ensure_dirs_exist(filename)    
    plt.savefig(filename)    

def step():
    """Commands executed at each time step.

    """
    global network, new_network

    # LOOK AT NEIGHBOURS AND GET RESOURCES PROBABILISTICALLY FROM NEIGHBOURS
    for n in network.nodes_iter():
        new_network = dfs(n, network, new_network, prob_visit)

    # FIND MATCHING PATTERN IN RESOURCE LISTS
    # TODO: This should also happen in the init code
    # TODO: Repeats below in 'step'. Move to a helper function?
    for n in network.nodes_iter():
        # print 'Node %s' %(n)
        for p in range(len(project_values_df.index)): 
            project_pattern = project_values_df.at[p,'project_code']
            # project_pattern = [1]#,2]
            matches = subfinder(network.node[n]['resources'],project_pattern)
            if matches:
                # print 'New pattern'
                # print network.node[n]['resources']
                network.node[n]['resources'] = subfilter(network.node[n]['resources'],project_pattern)
            else:
                pass
            #     print 'No Match'
            # print network.node[n]['resources']
            # print ''

    # UPDATE NETWORK TO NEW_NETWORK SINCE ALL ACTIONS FOR THIS TIME STEP ARE COMPLETE
    # TODO: Check if this is right. Could one be a shallow copy? network.copy() gives a deep copy. Maybe consider network, nextNetwork = nextNetwork, network
    network = new_network
    new_network = network

if __name__ == '__main__':
    """Main loop.

    """
    init()
    draw()
    
    for time in range(time_steps):
        step()
        draw()

    # TODO: Do some more checks. Add some feedback to make sure the code is right.
    # TODO: Add visualization of how network evolves.
    # TODO: Store output data and plot jobs over time.
    



