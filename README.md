# Analysing the impact of age, gender and ethnicity on the spread of viruses

Data driven approach to analyse the impact of biological cues, namely age, gender and ethnicity on the infection and recovery rates, with the use of compartmental modelling. We use COVID-19 as use case, due to the large amount of data currently available.

Using NetworkX we create a graph which reflects population, based on demographics from the data available from UK census, by setting attributes to each node, and then we run SIS (Susceptible-Infected-Susceptible) simulations on it. We then compare our results with the official figures provided by the UK government on the infections in the given population.

We compute infection rate based on currently available COVID-19 data, and infect each nodes based on its individual attributes. 

In addition to that we test VGG16 model to develop facial recognition system. This extracts age, gender and ethnicity from images, showing the potential for an infection tracking system with the use of images. Such system could use real-time image feed such as CCTV to monitor the population network and probability of the spread of the virus.
