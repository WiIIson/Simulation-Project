# Simulation_Project
 
By William Conley and Ryan De Sousa

## Instructions to run

The simulation code is contained in the file "Infection_Spread_Simulation.py". Run this file in order to see the simulation.

The simulation has parameters that have been set up to work, but can be adjusted to test different scenarios. Comments are included in the simulation to explain how the parameters can be adjusted.

Parameters that can be adjusted include:
- The number of people the simulation starts with, and their starting states
- Toggling the music on or off
- Showing visual overlays of how infection spreads, or how the people see each other
- Toggling the creation of a plot at the end of the simulation on or off
- Simulation constants (e.g., time step, max speed of each person, vision radius)

## Plotting

The file "InfectionPlots.py" contains code for creating a plot of the infection states over time, and was used to create the figure we compared to the mathematical model's results. After the simulation completes, you can run `df.to_csv("filename.csv")` in the console to save the results as a CSV file. This file can then be read into "InfectionPlots.py" in order to create the plot.

## Additional Files

This repository also contains additional files relevant to the project. The "FinalFigs" folder contains figures we included in our project, as well as the CSV data we used to create them. The "Reports" folder contains our project proposal, as well as a word document version of our report.

## References

This simulation was based on the following mathematical model:

Giordano, G., Blanchini, F., Bruno, R., Colaneri, P., Di Filippo, A., Di Matteo, A., & Colaneri, M. (2020). Modelling the COVID-19 epidemic and implementation of population-wide interventions in Italy. Nature Medicine, 26(6), 855–860. https://doi.org/10.1038/s41591-020-0883-7