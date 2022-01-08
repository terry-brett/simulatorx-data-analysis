import random
import pandas as pd
import cv2
from keras.models import load_model
import numpy as np

def read_file(filename):
    data = {}
    with open(filename, 'r+') as text:
        for line in text.readlines():
            key, value = line.strip().split(',')
            data[key] = value

    return data

def round_number(N, percentage):
    return round((float(percentage)/100.) *N)

def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def seed_graph(G, seeds):
    nodes = G.number_of_nodes()
    rand_nodes = random.sample(range(nodes), seeds)
    for n in range(0, nodes):
        if n in rand_nodes:
            G.nodes[n]['status'] = 'I'
        else:
            G.nodes[n]['status'] = 'S'

def power_law(x0, x1, gamma, N):
    pl = []
    for i in range (N):
        pl.append(((x1**(gamma+1) - x0**(gamma+1))*random.uniform(0,1)  + x0**(gamma+1.0))**(1.0/(gamma + 1.0)))
    return pl

def activate_graph(activity, N):
    active_nodes = []
    for i in range(N):
        if (random.random() < activity[i]):
            active_nodes.append(i)
    return active_nodes

def load_susceptibility_matrix():
    return pd.read_csv('../data/susceptibility_matrix.csv')

def load_recovery_matrix():
    return pd.read_csv('../data/recovery_matrix.csv')

def get_susceptibility_matrix_index(age):
    '''
        The age matrix is 16x16 and it's split in groups of 4,
        We can use whole division to quickly get the index
    '''
    if age >= 75:
        return 15
    else:
        return age//5

def get_recovery_rate(gender, age):
    nodes_recovery = load_recovery_matrix()

    if gender == 0:
        column = 'male'
    elif gender == 1:
        column = 'female'

    if age <= 19:
        return nodes_recovery[column][0]
    elif age >= 60:
        return nodes_recovery[column][5]
    else:
        return nodes_recovery[column][(age//10)-1]

def infection_rate(nodeA, nodeB, G, dataframe):
    ageA = G.nodes[nodeA]['age']
    ageB = G.nodes[nodeB]['age']
    gender = G.nodes[nodeA]['gender']
    ethnicity = G.nodes[nodeA]['ethnicity']

    row = get_susceptibility_matrix_index(ageA)
    col = get_susceptibility_matrix_index(ageB)

    age_infection_rate = dataframe.iloc[row, col]

    # infection probabilities for populations
    gender_infection_rate = [0.17, 0.146] # male, female infection rates
    population_infection_rate =[0.7392, 0.8618, 0.4927, 0.8799] # white, black, mixed, asian

    return age_infection_rate * gender_infection_rate[gender] * population_infection_rate[ethnicity]

def recovery_rate(node,G):
    age = G.nodes[node]['age']
    gender = G.nodes[node]['gender']
    ethnicity = G.nodes[node]['ethnicity']

    population_recover_rate = [0.1585, 0.2910, 0.1923, 0.1585] # white, black, mixed, asian

    gender_recover_rate = get_recovery_rate(gender, age)

    return gender_recover_rate * population_recover_rate[ethnicity]

def infect(active_nodes, G, dataframe):
    for n in active_nodes:
        neighbors = list(G.neighbors(n))
        if len(neighbors) > 0:
            for neighbor in neighbors:
                if G.nodes[n]['status'] == 'S':
                    if random.uniform(0,1) < (infection_rate(n, neighbor, G, dataframe) - 0.076):
                        G.nodes[n]['status'] = 'I'

def recover(active_nodes, G):
    for n in active_nodes:
        if G.nodes[n]['status'] == 'I':
            if random.uniform(0,1) < recovery_rate(n, G):
                G.nodes[n]['status'] = 'S'

def count_compartament_data(G):
    dod = {} # dict of dicts
    for node in G.nodes:
        dod[node] = G.nodes[node]

    df = pd.DataFrame(dod).transpose() # swap rows and columns
    status_counts = df['status'].value_counts()

    return status_counts.S, status_counts.I # return the number of susceptible and infected

def count_attribute(G, attribute=None, infected=False, ethinicty=None, gender=None, age=None):
    # TODO: Started working on a function that counts population based on any attribute

    dod = {}  # dict of dicts
    for node in G.nodes:
        dod[node] = G.nodes[node]

    df = pd.DataFrame(dod).transpose()  # swap rows and columns

    # set age boundries for easy count of node attributes
    age_bins = [1, 14, 15, 54, 55, 110]
    age_boundry = age_bins[0:]
    if age is not None:
        if age == 'youth':
            age_boundry = age_bins[0:2]
        elif age == 'adult':
            age_boundry = age_bins[2:4]
        elif age == 'senior':
            age_boundry = age_bins[4:]

    df['binned'] = pd.cut(df['age'], age_bins)

def make_prediction(image_path):
    img_arr = []
    path_list = []

    if image_path.endswith(".jpg") or image_path.endswith(".jpeg") or image_path.endswith(".JPG"):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (96, 96))
        img_arr.append(image)
        path_list.append(image_path)

    img_arr = np.asarray(img_arr)
    img_arr = img_arr / 255.0

    eth_model = load_model('../data/facial_features/eth-model.h5')
    gen_model = load_model('../data/facial_features/gen-model.h5')
    age_model = load_model('../data/facial_features/age-model.h5')

    eth_pred = eth_model.predict(img_arr)
    gen_pred = gen_model.predict(img_arr)
    age_pred = age_model.predict(img_arr)

    eth_list = np.argmax(eth_pred, axis=1)
    gen_list = np.argmax(gen_pred, axis=1)
    age_list = np.argmax(age_pred, axis=1)

    data = {
        'eth': int(eth_list[0]),
        'gen': int(gen_list[0]),
        'age': int(age_list[0])
    }

    print (data)