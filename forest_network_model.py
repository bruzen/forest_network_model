# from __future__ import print_function
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
    """Remove values matching a sublist from a list.

    Args:
        resource_list (list): the list to filter sublist values from.
        project_pattern (list): the sublist to look for.
    Returns:
        a list with characters from the sublist removed.

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
    considered, stack = set(), [start]
    while stack:
        # Pop a node index from the stack
        vertex = stack.pop()
        # Consider visiting the node if unconsidered
        if vertex not in considered:
            considered.add(vertex)
            # Actually visit the node if a random number is greater than some threshold
            if visit_prob > random.random():
                # Add all the 'vertex' node's neighbours to the stack to consider visiting
                stack.extend(set(graph[vertex].keys()) - considered)
                # Copy resources from the 'vertex' node to the 'start' node
                new_graph.node[start]['resources'] = new_graph.node[start]['resources'] + graph.node[vertex]['resources']
                # Remove duplicates from 'start' node's resources list and sort list
                new_graph.node[start]['resources'] = sorted(list(set(new_graph.node[start]['resources']))) 
    return new_graph

def init(params):
    """Initialization code runs at start of simulation.

    """ 
    global time, time_steps, done, tree, height, branching, nodes_for_small_world, k, p, prob_visit, depth, find, forget, network, new_network, positions, prob_visit, project_values_df, total_opportunities_found, total_jobs_created, cum_jobs_not_found, time_all_jobs_found, opportunities_found_df, max_no_jobs
    # TODO: Some of these are unused. Remove them (e.g. depth, find, forget?).

    # Experiment parameters, defined as part of experiment and so commented out below
    p           = params['prob_rewiring']
    prob_visit  = params['prob_visiting']
    tree        = params['tree']
    
    # Time parameters
    time                = 0
    time_steps          = 1000
    done                = False # Don't perform a search for opportunities if all possible jobs have been found

    # Tree parameters
    ### tree        = False     # If True make a tree, else make a small world network
    branching   = 5             # Branching factor of the tree. Should be odd so k = branching + 1 is even
    height      = 4             # Height of the tree

    temp_tree = nx.balanced_tree(branching,height) # Temp tree used only to compute nodes for small world network

    # Small world network parameters (small world network created if Tree == False)
    nodes_for_small_world   = temp_tree.number_of_nodes()     # The number of nodes, same as for tree
    k                       = branching + 1 # Each node is connected to k nearest neighbors in ring topology
    ### p                       = 0.03      # The probability of rewiring each edge
    tries                   = 100           # Number of attempts to generate a connected graph
    seed                    = None          # Seed for random number generator (default=None)

    # Algorithm parameters
    ### prob_visit          = 0.10          # Probability of visiting each node

    # Results and output data
    total_opportunities_found   = 0
    total_jobs_created          = 0
    opportunities_found_df      = pd.DataFrame(columns=('Opportunities Found', 'Jobs Created', 'Loss Measure')) # Table of results stored as a pandas DataFrame
    cum_jobs_not_found          = 0.0 # Cummulative jobs not found. A measure of efficiency. Value from end of run used.
    time_all_jobs_found         = int(time_steps + time_steps/5) # Time at which all jobs are found in a run. time_all_jobs_found

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
    res_len_check = 0   # Error checking maybe no longer needed
    df_loc = 0          # An index
    for key, val in reward_dist.items():
        project_val = key
        for project_length in range(len(val)):
            project_code = resources_list_temp[:val[project_length]]
            del resources_list_temp[:val[project_length]]
            res_len_check += val[project_length]
            project_values_df.loc[df_loc] = [project_val, project_code] #[random.randint(-1,1) for n in range(2)]
            df_loc +=1
    # project_values_df.loc[df_loc] = [project_val, [1]] # TESTING CODE SO SEARCH ALWAYS FINDS SOMETHING (SOMEONE ALWAYS HAS RESOURCE '1'). DELETE EVENTUALLY.   

    max_no_jobs = project_values_df['project_val'].sum()

    # INITIALIZE NETWORK AND NODE ATTRIBUTES (RESOURCES AND JOBS/SCORE)
    if tree == True: # Create a tree
        network = nx.balanced_tree(branching,height)
    else: # Create a small world network 
        network = nx.connected_watts_strogatz_graph(nodes_for_small_world, k, p, tries, seed) # Change just this line to change graph type/construction
    positions = nx.circular_layout(network) # Change just this line to change network graph layout

    for n in network.nodes_iter():
        network.node[n]['resources']   = []
        network.node[n]['jobs']        = 0
        network.node[n]['found_count'] = 0

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

    # MAKE A COPY OF THE NETWORK FOR THE NEXT TIMESTEP
    new_network = network.copy()

def draw():
    """Plotting code runs at start of simulation and after each time step.

    """
    plt.cla()

    # For plotting, try generating a dictinary with labels for each node
    labels_list=dict((n,d['resources']) for n,d in network.nodes(data=True))

    res = [100 * network.node[i]['jobs'] for i in network.nodes_iter()]
    nx.draw_networkx(network,positions,edgelist=network.edges(),
        node_size=res,with_labels=True, labels=labels_list)
    plt.axis('image')
    plt.title('Network Information and Jobs at Time %i' %(time))
    filename = 'Plots/Time_step_' + str(time) + '.png'
    ensure_dirs_exist(filename)    
    plt.savefig(filename)    

def step():
    """Commands executed at each time step.

    """
    global done, network, new_network, total_opportunities_found, total_jobs_created, cum_jobs_not_found, time_all_jobs_found

    if not done:
        # LOOK AT NEIGHBOURS AND GET RESOURCES PROBABILISTICALLY FROM NEIGHBOURS
        for n in network.nodes_iter():
            new_network = dfs(n, network, new_network, prob_visit)

        # FIND MATCHING PATTERN IN RESOURCE LISTS
        # TODO: This should also happen in the init code
        # TODO: Repeats below in 'step'. Move to a helper function?
        for n in network.nodes_iter():
            for p in range(len(project_values_df.index)): 
                project_pattern = project_values_df.at[p,'project_code']
                # project_pattern = [1]#,2]
                matches = subfinder(network.node[n]['resources'],project_pattern)
                if matches:
                    # TODO: Should this update the new_network, should it access the new network?
                    # TODO: Now we can only find opportunity in one turn. What if many nodes find it at the same time? Change so we only count once? Maybe select randomly which version to keep? Fixed for now by deleting resources if found once.
                    
                    # IF MATCHES, DELETE FOR ALL NODES SO THOSE VALUES NO LONGER GET PASSED ALONG
                    for nn in network.nodes_iter():
                        new_network.node[nn]['resources'] = subfilter(new_network.node[nn]['resources'],project_pattern)
                    
                    # STORE DATA ON OPPORTUNITIES FOUND AND JOBS CREATED
                    new_network.node[n]['jobs'] += project_values_df.at[p,'project_val']
                    total_jobs_created          += project_values_df.at[p,'project_val']
                    new_network.node[n]['found_count']  += 1
                    total_opportunities_found           += 1

                    # # PRINT RESULTS TO TERMINAL AS FEEDBACK. COMMENT OUT OR DELETE LATER.
                    # print ''
                    # print 'Time step: %i. Found something:' %(time)
                    # print matches
                    # print 'Node: %i' %(n)
                    # print 'Opportunities found by node so far: %i' %(new_network.node[n]['found_count'])
                    # print 'Jobs created by this node so far: %i' %(new_network.node[n]['jobs'])
                    # print 'Total jobs creates so far: %i' %(total_jobs_created)
        cum_jobs_not_found += (max_no_jobs - total_jobs_created)
        # print 'Time: %i, Cum jobs not found: %s, max_no_jobs: %s, total_jobs_created: %s' %(time, cum_jobs_not_found,max_no_jobs,total_jobs_created) 
        if (max_no_jobs - total_jobs_created) == 0:
            done = True
            time_all_jobs_found = time
    else: # if done, don't bother computing matches since no more opportunities can be found
        # TODO: contribution to cum_jobs_n_found should be zero if done. Else throw error.
        pass
    opportunities_found_df.loc[time] = [total_opportunities_found, total_jobs_created, cum_jobs_not_found]

    # UPDATE NETWORK TO NEW_NETWORK SINCE ALL ACTIONS FOR THIS TIME STEP ARE COMPLETE
    # TODO: Check if this is right. Could one be a shallow copy? network.copy() gives a deep copy. Maybe consider network, nextNetwork = nextNetwork, network
    network = new_network
    new_network = network

# def final_draw():
#     """Plotting and analaysis executed at the end of the run.

#     """
#     plt.cla()
#     df_jobs_ops = opportunities_found_df[['Opportunities Found', 'Jobs Created']]
#     plt.figure(); df_jobs_ops.plot(); plt.legend(loc='best')
#     plt.title('Jobs and Opportunities Found Over Time')
#     plt.xlabel('Time')
#     plt.ylabel('Count')
#     filename = 'Plots_Final/Opportunities_and_Jobs_vs_Time_' + str(nodes_for_small_world) + '.png'
#     ensure_dirs_exist(filename)    
#     plt.savefig(filename)      

#     plt.cla()
#     df_eff = opportunities_found_df[['Loss Measure']]
#     plt.figure(); df_eff.plot(); plt.legend(loc='best')
#     plt.title('Loss Measure Over Time')
#     plt.xlabel('Time')
#     plt.ylabel('Loss Measure')
#     filename = 'Plots_Final/Cum_Jobs_Not_Found_Measure_vs_Time_' + str(nodes_for_small_world) + '.png'
#     ensure_dirs_exist(filename)    
#     plt.savefig(filename) 


def run_simulation(params):
    global time
    init(params)
    # draw()
    for time in range(time_steps):
        step()
        # draw()
    time_measure = time_all_jobs_found
    loss_measure = cum_jobs_not_found
    return loss_measure, time_measure

if __name__ == '__main__':
    """Main loop.

    """
    # EXPERIMENT PARAMETERS
    runs                 = 5
    # experiments          = {'Probability_of_Rewiring': [0.05, 0.1, 0 0.2, .3, .4, .5, .6], 'Probability_of_Visiting': [0.05, 0.1, 0.2, .3, .4, .5, .6]}
    experiments          = {'Probability_of_Visiting': [0.06, 0.1,0.14,0.18,0.22, 0.26, .3]}
    measures             = ['loss_measure', 'time_measure']
    networks             = ['Tree', 'Small_World']
    # Default parameter values when parameter not being tested
    default_prob_rewiring     = 0.1 
    default_prob_visiting     = 0.15

    params = {'Probability_of_Rewiring': default_prob_rewiring, 'prob_visiting': default_prob_visiting, 'tree': True}
    for experiment, experiment_name in enumerate(experiments):
        print ''
        for run in range(runs):
            print 'run: %i' %(run) 
            df_loc = 0
            df_loss_measure = pd.DataFrame(columns=[networks])
            df_time_measure = pd.DataFrame(columns=[networks])
            for param_index, param_val in enumerate(experiments[experiment_name]):
                print '%s: %f' %(experiment_name, experiments[experiment_name][param_index])
                loss_measure_row = []
                time_measure_row = []
                for network, network_name in enumerate(networks):
                    if experiment_name == 'Probability_of_Rewiring':
                        if network_name == 'Tree':
                            params = {'prob_rewiring': experiments['Probability_of_Visiting'][param_index], 'prob_visiting': default_prob_visiting, 'tree': True}
                        elif network_name == 'Small_world':
                            params = {'prob_rewiring': experiments['Probability_of_Visiting'][param_index], 'prob_visiting': default_prob_visiting, 'tree': False}
                        else:
                            pass # TODO: Throw value error. This should be impossible
                    elif experiment_name == 'Probability_of_Visiting':
                        if network_name == 'Tree':
                            params = {'prob_rewiring': default_prob_rewiring, 'prob_visiting': experiments['Probability_of_Visiting'][param_index], 'tree': True}
                        elif network_name == 'Small_World':
                            params = {'prob_rewiring': default_prob_rewiring, 'prob_visiting': experiments['Probability_of_Visiting'][param_index], 'tree': False}
                        else:
                            pass # TODO: Throw value error. This should be impossible
                    else:
                        pass # TODO: Throw value error. This should be impossible                            
                    loss_measure, time_measure = run_simulation(params)
                    loss_measure_row.append(loss_measure) 
                    time_measure_row.append(time_measure)

                df_loss_measure.loc[df_loc] = loss_measure_row    # project_values_df.loc[df_loc] = [project_val, project_code]
                df_time_measure.loc[df_loc] = time_measure_row
                df_loc += 1

            df_loss_measure['Param'] = experiments[experiment_name]
            df_time_measure['Param'] = experiments[experiment_name]

            print df_loss_measure
            print df_time_measure
            print ''

            loss_measure_filename = 'Data/' + 'Loss_Measure-' + str(experiment_name) + '-Runs' + str(runs) + '-Nodes' + str(nodes_for_small_world) + '-Height' + str(height) + '-k' + str(k) + '-p' + str(p) + '-TimeSteps' + str(time_steps) + '/Loss_Measure-' + str(experiment_name) +'-' + 'Run_' + str(run) + '.csv'
            ensure_dirs_exist(loss_measure_filename)
            df_loss_measure.to_csv(loss_measure_filename)
            
            time_measure_filename = 'Data/' + 'Time_to_Completion-' + str(experiment_name) +'-Runs' + str(runs) + '-Nodes' + str(nodes_for_small_world) + '-Height' + str(height) + '-k' + str(k) + '-p' + str(p) + '-TimeSteps' + str(time_steps) + '/Time_to_Completion-' + str(experiment_name) +'-' + 'Run_' + str(run) + '.csv'
            ensure_dirs_exist(time_measure_filename)
            df_time_measure.to_csv(time_measure_filename)

    # TODO: Do some more checks. Add some feedback to make sure the code is right.
    # TODO: Add visualization of how network evolves.
    # TODO: Store output data and plot jobs over time.
    # NOTE: If resources not deleted, algorithm can find the same opportunity many times.




