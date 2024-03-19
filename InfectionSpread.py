import pygame
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Simulation display colours
C_S = (150, 150, 255)
C_IS1 = (255, 100, 100)
C_IS2 = (200, 0, 0)
C_IS3 = (100, 0, 0)
C_D = (0, 0, 0)
C_H = (0, 150, 0)
C_HOSP = (200, 200, 0)

# Simulation screen size
win_width = 640
win_height = 640

# Radius of infection
INFECT_RADS = {'S': 0,
               'I': 25,
               'D': 25,
               'A': 50,
               'R': 50,
               'T': 0,
               'H': 0,
               'E': 0}

SIM_ALPHA   = 0.570
SIM_BETA    = 0.011
SIM_DELTA   = 0.011
SIM_GAMMA   = 0.456
SIM_EPSILON = 0.171
SIM_THETA   = 0.371
SIM_ZETA    = 0.125
SIM_ETA     = 0.125
SIM_MU      = 0.017
SIM_NU      = 0.027
SIM_TAU     = 0.010
SIM_LAMBDA  = 0.034
SIM_RHO     = 0.034
SIM_KAPPA   = 0.017
SIM_XI      = 0.017
SIM_SIGMA   = 0.017

# Return boolean based on a probability
def randProb(prob):
    return random.random() < prob

# Calculate direction vector for given angle and magnitude
def calcDirVec(theta, mag):
    x = mag * np.cos(theta)
    y = mag * np.sin(theta)
    return np.array([x, y])

# Class for people in the simulation
class Person:
    # Initialize entity
    def __init__(self, x, y, istat, vision, speed):
        self.x = x              # Person's x coordinate
        self.y = y              # Person's y coordinate
        self.istat = istat      # Person's infection status
        self.vision = vision    # Person's field of view
        self.speed = speed      # Person's speed
        self.maxSpeed = speed
        self.irad = INFECT_RADS[self.istat]         # Radius of infection
        self.theta = random.random() * 2 * np.pi    # Person's angle
        self.time_until_S = 60
        self.time_H = 0
    
    # Determine which people this person has seen
    def seen(self, people):
        
        pseen = []
        
        for person in people:
            sdist = np.array([self.x, self.y])
            odist = np.array([person.x, person.y])
            
            if np.sqrt(np.sum((sdist - odist)**2)) <= self.vision:
                pseen.append(person)
        
        return pseen
    
    def inIrad(self, people):
        
        inIrad = []
        
        for person in people:
            sdist = np.array([self.x, self.y])
            odist = np.array([person.x, person.y])
            
            if np.sqrt(np.sum((sdist - odist)**2)) <= INFECT_RADS[person.istat]:
                inIrad.append(person)
        
        return inIrad
    
    # Update the person's location and state
    def update(self, people, dt):
        
        # Get list of people the person is able to see
        seen = self.seen(people)
        inIrad = self.inIrad(people)
        
        # Update state
        # 1. If not infected, check for infected people
        # 2. If infected, calculate chance to advance stage
        
        if (self.istat == "S"):
            for person in inIrad:
                # Random chance to become infected
                if person.istat == "I":
                    if randProb(SIM_ALPHA * dt / 10):
                        self.istat = "I"
                elif person.istat == "D":
                    if randProb(SIM_ALPHA * dt / 10):
                        self.istat = "I"
                elif person.istat == "A":
                    if randProb(SIM_GAMMA * dt / 10):
                        self.istat = "I"
                elif person.istat == "R":
                    if randProb(SIM_GAMMA * dt / 10):
                        self.istat = "I"
                        
        elif (self.istat == "I"):
            # random chance to progress infection
            if randProb(SIM_ZETA * dt / 10):
                self.istat = "A"
            # random chance to become detected
            for person in inIrad:
                if randProb(SIM_EPSILON * dt / 10):
                    self.istat = "D"
            # random chance to heal
            if randProb(SIM_LAMBDA * dt / 10):
                self.istat = "H"
                self.time_H = self.time_until_S
                    
        elif (self.istat == "D"):
            # random chance to progress infection
            if randProb(SIM_ETA * dt / 10):
                self.istat = "R"
            # random chance to heal
            if randProb(SIM_RHO * dt / 10):
                self.istat = "H"
                self.time_H = self.time_until_S
        
        elif (self.istat == "A"):
            # random chance to progress infection
            if randProb(SIM_MU * dt / 10):
                self.istat = "T"
                self.speed *= 0.5
            # random chance to become detected
            for person in inIrad:
                if randProb(SIM_THETA * dt / 10):
                    self.istat = "R"
            # random chance to heal
            if randProb(SIM_KAPPA * dt / 10):
                self.istat = "H"
                self.time_H = self.time_until_S
                    
        elif (self.istat == "R"):
            # random chance to progress infection
            if randProb(SIM_NU * dt / 10):
                self.istat = "T"
                self.speed *= 0.5
            # random chance to heal
            if randProb(SIM_XI * dt / 10):
                self.istat = "H"
                self.time_H = self.time_until_S
        
        elif (self.istat == "T"):
            # random chance to progress infection
            if randProb(SIM_TAU * dt / 10):
                self.istat = "E"
            # random chance to heal
            if randProb(SIM_SIGMA * dt / 10):
                self.speed = self.maxSpeed
                self.istat = "H"
                self.time_H = self.time_until_S
        
        
        
        
        # Move person
        if (self.istat != "E"):
            
            # Count down timer until susceptible again
            if self.istat == "H":
                self.time_H -= dt
                if self.time_H <= 0:
                    self.istat = "S"
        
            # Direction is a comination of multiple vectors:
            # 1. Random direction the person wants to go
            #    with weight 1
            # 2. Vector going away from seen infected individuals
            #    weighted by distance from the individual
            #
            
            # Overall direction vector
            dirVec = np.zeros(2)
            
            # Calculate random direction vector
            newTheta = self.theta + (random.random()) - (0.5)
            randDirVec = calcDirVec(newTheta, self.speed)
            
            avoidVec = np.zeros(2)
            
            for person in seen:
                if person.istat in ["D", "R", "T"]:
                    selfPosVec = np.array([self.x, self.y])
                    otherPosVec = np.array([person.x, person.y])
                    pdist = selfPosVec - otherPosVec
                    uDist = self.speed * (pdist / np.linalg.norm(pdist))
                    
                    # Weight by distance
                    
                    avoidVec += uDist
            
            # Add vectors together and normalize
            dirVec += 0.1 * randDirVec
            dirVec += avoidVec
            
            # Correct dirVec if person cannot go off screen
            if ((self.x == 0 and dirVec[0] < 0) or (self.x == win_width and dirVec[0] > 1)) and np.linalg.norm(avoidVec) > 0:
                dirVec[0] = 0
            if ((self.y == 0 and dirVec[1] < 0) or (self.y == win_height and dirVec[1] > 1)) and np.linalg.norm(avoidVec) > 0:
                dirVec[1] = 0
            
            if np.linalg.norm(dirVec) != 0:
                dirVec = self.speed * (dirVec / np.linalg.norm(dirVec))
            
            # Update position
            self.x += dirVec[0] * dt
            self.y += dirVec[1] * dt
        
            # Correct if the person goes off screen
            if self.x < 0:
                self.x = 0
                self.theta = -self.theta + np.pi
            if self.x > win_width:
                self.x = win_width
                self.theta = -self.theta + np.pi
            if self.y < 0:
                self.y = 0
                self.theta *= -1
            if self.y > win_height:
                self.y = win_height
                self.theta *= -1

# Class controlling the simulation
class Simulation:
    def __init__(self, vision, speed, width, height):
        # Simulation variables that are the same for each person
        self.vision = vision    # Max vision of each person
        self.speed = speed      # Max speed of each person
        self.people = []        # People controlled by the simulation
        self.t = 0              # Simulation time
        self.dt = 0.3           # Simulation time step
        self.see_vrad = False   # Toggle for seeing the radius around the entity
        self.see_irad = False   # Toggle for seeing the infection radius around the person
        self.see_seen = False   # Toggle for seeing lines between entities that see each other
        self.see_iseen = False  # Toggle for seeing lines of infection
        self.person_rad = 10    # Display size of people
        self.width = width
        self.height = height
        self.state = {}
        
    # Toggle functions    
    
    # Toggle vision radius
    def set_vrad(self, newRad):
        self.see_vrad = newRad
        
    # Toggle vision lines
    def set_vseen(self, newSeen):
        self.see_seen = newSeen
    
    def set_irad(self, newRad):
        self.see_irad = newRad
        
    def set_iseen(self, newSeen):
        self.see_iseen = newSeen
    
    # Add a new person to the simulation
    def add_person(self, x, y, istat):
        e = Person(x, y, istat, self.vision, self.speed)
        self.people.append(e)
    
    # Add a person to a random location in the simulation
    def add_rand_person(self, istat):
        x = random.random() * self.width
        y = random.random() * self.height
        e = Person(x, y, istat, self.vision, self.speed)
        self.people.append(e)
    
    # Calculate array of which person is allowed to heal
    def calcHealPerms(self):
        allocated = False
        perms = np.full(len(self.people), False)
        for i in range(len(self.people)):
            person = self.people[i]
            if (person.istat == "T" and allocated == False):
                perms[i] = True
                allocated = True
                
    
    # Draw the current state of the simulation
    def draw(self, screen):
        
        for person in self.people:
            
            # Draw vision radius
            if self.see_vrad:
                pygame.draw.circle(screen, BLACK, (person.x, person.y), self.vision, 1)
            
            # Draw infection radius
            if self.see_irad:
                pygame.draw.circle(screen, BLACK, (person.x, person.y), INFECT_RADS[person.istat], 1)
            
            # Draw vision lines
            if self.see_seen:
                eseen = person.seen(self.people)
                for person2 in eseen:
                    pygame.draw.line(screen, BLACK, (person.x, person.y), (person2.x, person2.y))
            
            # Draw infection lines
            if self.see_iseen:
                eseen = person.inIrad(self.people)
                for person2 in eseen:
                    pygame.draw.line(screen, C_IS1, (person.x, person.y), (person2.x, person2.y))
            
            # Draw people
            if (person.istat=='S'):
                pygame.draw.circle(screen, C_S, (person.x, person.y), self.person_rad, 0)
            elif (person.istat=='I'):
                pygame.draw.circle(screen, C_IS1, (person.x, person.y), self.person_rad, 0)
                pygame.draw.circle(screen, C_S, (person.x, person.y), self.person_rad, 4)
            elif (person.istat=='D'):
                pygame.draw.circle(screen, C_IS1, (person.x, person.y), self.person_rad, 0)
            elif (person.istat=='A'):
                pygame.draw.circle(screen, C_IS2, (person.x, person.y), self.person_rad, 0)
                pygame.draw.circle(screen, C_S, (person.x, person.y), self.person_rad, 4)
            elif (person.istat=='R'):
                pygame.draw.circle(screen, C_IS2, (person.x, person.y), self.person_rad, 0)
            elif (person.istat=='T'):
                pygame.draw.circle(screen, C_IS3, (person.x, person.y), self.person_rad, 0)
            elif (person.istat=='H'):
                pygame.draw.circle(screen, C_H, (person.x, person.y), self.person_rad, 0)
            elif (person.istat=='E'):
                pygame.draw.circle(screen, C_D, (person.x, person.y), self.person_rad, 0)
    
    # Update the entities in the simulation and the simulation state
    def update(self, screen):
        
        newState = {"S":0,
                    "I":0,
                    "D":0,
                    "A":0,
                    "R":0,
                    "T":0,
                    "H":0,
                    "E":0}
        
        # Update each individual person
        for i in range(len(self.people)):
            newState[self.people[i].istat] += 1
            # Pass list of people subtracting current person
            e_update = self.people.copy()
            del e_update[i]
            self.people[i].update(e_update, self.dt)
        
        # Update simulation state
        self.state = newState



# Set up simulation data
pygame.init()
win_width = 640
win_height = 640
screen = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption('Pandemic Simulation')
clock = pygame.time.Clock()

# Set up simulation entities
sim = Simulation(100, 10, win_width, win_height)

for i in range(50):
    sim.add_rand_person("S")

for i in range(3):
    sim.add_rand_person("I")
#sim.set_irad(True)
#im.set_iseen(True)
sim.set_vseen(True)

# Load music
pygame.mixer.init()
pygame.mixer.music.load('PlagueInc.mp3')
pygame.mixer.music.play(-1)

df = pd.DataFrame(columns=['S','I','D','A','R','T','H','E'])

makePlot = False

while (True):
    clock.tick(30)
    
    event = pygame.event.poll()
    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
        pygame.quit()
        break
    
    # Fill background
    screen.fill(WHITE)
    
    # Update simulation and draw
    sim.update(screen)
    sim.draw(screen)
    if makePlot:
        df.loc[len(df)] = list(sim.state.values())
    
    # Update screen
    pygame.display.flip()

if makePlot:
    df.plot.area()