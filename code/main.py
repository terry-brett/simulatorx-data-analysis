from simulations import *
import glob
import os
from data_download import *

'''
    Run simulations for given populations, by adding age, gender and ethnicity as attributes and compare with GOV.UK data
'''

list_of_areas = ["westminster", "enfield", "greenwich", "redbridge"]

# NOTE: change results folder in Code Ocean, line 123 in simulations.py

for area_name in list_of_areas:
    data = read_file("../data/" + area_name + "_demographics.csv")

    ks2_test = run_simulation(data, area_name)
    print("Kolmogorov–Smirnov test for", area_name.capitalize(), ":")
    print (ks2_test)

''' 
    Facial recognition simulations that extract age, gender and ethnicity and assign those as node attributes
'''

# check if pre-trained models exist if not download them
model_files = ["age-model.h5", "eth-model.h5", "gen-model.h5"]

path = '../data/facial_features'
found_files = [f for f in os.listdir(path) if f.endswith('.h5')]

not_found = list(set(model_files) - set(found_files))
for file in not_found:
    download(file, path)

list_of_images = glob.glob("../data/facial_images/*")

for image in list_of_images:
    make_prediction(image)
