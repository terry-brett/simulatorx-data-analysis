from simulations import *

list_of_area = ["westminster", "enfield", "greenwich", "redbridge"]

area_name = "westminster"
data = read_file("../data/" + area_name + "_demographics.csv")

ks2_test = run_simulation(data, area_name)
print("Kolmogorov–Smirnov test for", area_name.capitalize(), ":")