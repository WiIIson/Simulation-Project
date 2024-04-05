import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('FinalFigs/Trial1.csv')
dt = 0.3
maxPeople = 53

plt.plot(data['Unnamed: 0'] * dt, data['I'] / maxPeople, 'b')
plt.plot(data['Unnamed: 0'] * dt, data['D'] / maxPeople, 'c')
plt.plot(data['Unnamed: 0'] * dt, data['A'] / maxPeople, 'g')
plt.plot(data['Unnamed: 0'] * dt, data['R'] / maxPeople, 'm')
plt.plot(data['Unnamed: 0'] * dt, data['T'] / maxPeople, 'orange')
plt.legend(['I (ND AS)', 'D (D AS)', 'A (ND S)', 'R (D S)', 'T (D IC)'])
plt.ylabel('Percent of population')
plt.xlabel('Time (s)')
plt.title('Percentages of infected subpopulations over time')
plt.show()