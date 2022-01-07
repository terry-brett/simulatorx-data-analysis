import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.stats import ttest_ind,kstest,ks_2samp
from helper import *

def run_simulation(data, area_name):
    N = int(data.get('total_population'))

    white_population = round_number(N, data.get('white_population'))
    black_population = round_number(N, data.get('black_population'))
    mixed_population = round_number(N, data.get('mixed_population'))
    asian_population = round_number(N, data.get('asian_population'))


    male_population = round_number(N, data.get('male_population'))
    female_population = round_number(N, data.get('female_population'))


    youth_population = round_number(N, data.get('youth_population'))
    adult_population = round_number(N, data.get('adult_population'))
    senior_population = round_number(N, data.get('senior_population'))

    seed = int(data.get('seed'))

    # generate graph

    G = nx.Graph()

    G.add_nodes_from(np.arange(0,N))

    # set all nodes to susceptible
    nx.set_node_attributes(G,'S','status')

    nodes_list = list(G.nodes)
    random.shuffle(nodes_list) # randomize the list of nodes

    # partition the list of nodes into uneven list of lists
    partitions = [white_population, black_population, mixed_population, asian_population]
    nodes_parts = np.split(nodes_list, np.cumsum(partitions))

    # split the graph population into 4 ethnic groups
    wn_dict = {n : {'ethnicity': 0} for n in nodes_parts[0]} # 0 - white_population
    bn_dict = {n : {'ethnicity': 1} for n in nodes_parts[1]} # 1 - black_population
    mn_dict = {n : {'ethnicity': 2} for n in nodes_parts[2]} # 2 - mixed_population
    an_dict = {n : {'ethnicity': 3} for n in nodes_parts[3]} # 3 - asian_population

    male_n, female_n = partition(nodes_list, 2)

    male_dict = {n : {'gender': 0} for n in male_n} # 0 - male_population
    female_dict = {n : {'gender': 1} for n in female_n} # 1 - female_population

    youth_n, adult_n, senior_n = partition(nodes_list, 3)

    '''
        Generate age for the population such that age:
        youth 1 < age < 14,
        adult 15 < age < 54,
        senior 55 < age < 110
    '''

    youth_dict = {n : {'age': random.randint(1,14)} for n in youth_n}
    adult_dict = {n : {'age': random.randint(15,54)} for n in adult_n}
    senior_dict = {n : {'age': random.randint(55,110)} for n in senior_n}

    # set generated attributes
    nx.set_node_attributes(G, wn_dict)
    nx.set_node_attributes(G, bn_dict)
    nx.set_node_attributes(G, mn_dict)
    nx.set_node_attributes(G, an_dict)

    nx.set_node_attributes(G, male_dict)
    nx.set_node_attributes(G, female_dict)

    nx.set_node_attributes(G, youth_dict)
    nx.set_node_attributes(G, adult_dict)
    nx.set_node_attributes(G, senior_dict)

    seed_graph(G, seed) # seed the graph based on inital number of infection within the given population

    # use activity graph, meaning we connect nodes at random at each timestep
    epsilon = 0.001
    eta = 1.
    gamma = -2.1
    act = power_law(epsilon, 1, gamma, N) # create node activity distribution

    node_susceptibility = load_susceptibility_matrix()

    sus = []
    inf = []

    # run SIS simulation for 113 timesteps
    for t in range(113):
        active_nodes = activate_graph(act, N)
        for node in active_nodes:
            count = 0
            while count < 2:
                target = random.randint(0, N - 1)
                if target != node and target not in G.neighbors(node):
                    G.add_edge(node, target)
                    count += 1
        infect(active_nodes, G, node_susceptibility)
        recover(active_nodes, G)
        S, I = count_compartament_data(G)
        print (str(t) + ", " + str(I))
        sus.append(S)
        inf.append(I)

    gov_uk_data = pd.read_csv('../data/' + area_name + '_gov_uk_data.csv',index_col=0)
    dff = pd.DataFrame()
    dff['NHS'] = gov_uk_data['Cases']
    dff = dff.iloc[::-1]
    dff['SIS'] = inf

    plt.plot(dff)
    plt.title('SIS v/s NHS for ' + area_name.capitalize(), fontsize=15)
    plt.xlabel('Days')
    plt.ylabel('Cases')
    plt.legend(dff.columns)
    plt.savefig('/results/test.png')

    return ks_2samp(dff['NHS'],dff['SIS'])