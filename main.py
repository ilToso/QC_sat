
import os
import copy
import math
import time
import pandas as pd
import numpy as np
import random as rd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import dwavebinarycsp
import dwave.inspector
from dwave.samplers import TabuSampler, SteepestDescentSolver, SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dimod import ConstrainedQuadraticModel, Binary, quicksum, ExactCQMSolver, BinaryQuadraticModel

rd.seed(0)
np.random.seed(0)

DWAVE_API_TOKEN = 'DEV-98e5734b73477c9d811c0485afa3c09067bb0952'

def bidim_gauss(x, y, c_x, c_y, A=1, sigma_x=1, sigma_y=1):
    exponent_term = -((x - c_x)**2 / (2 * sigma_x**2)) - ((y - c_y)**2 / (2 * sigma_y**2))
    return A * np.exp(exponent_term)


def generate_colors(num_colors):
    # Utilizza la mappa di colori 'viridis'
    colormap = plt.cm.get_cmap('viridis', num_colors)

    # Crea un array di valori da 0 a 1 per rappresentare la transizione del colore
    color_values = np.linspace(0, 1, num_colors)

    # Converte i valori della mappa di colori in colori effettivi
    colors = [colormap(value) for value in color_values]

    return colors

def plot_trajectories(epochs, n_sat, time_dictionary, name_experiment):
    for ep in range(epochs):
        if ep % 20 == 0 or ep == epochs-1:
            plt.figure(figsize=(6, 6))
            if ep == 0:
                for node in range(n_sat):
                    temp_list_x = time_dictionary['sat_'+str(node)+'_x']
                    temp_list_y = time_dictionary['sat_'+str(node)+'_y']

                    plt.plot(temp_list_x[0], temp_list_y[0], 'o', color='red')

            else:
                for node in range(n_sat):
                    plt.plot(time_dictionary['sat_'+str(node)+'_x'][:ep+1], time_dictionary['sat_'+str(node)+'_y'][:ep+1], '-', color=col_lines[int(node)], alpha=0.5)  
                for node in range(n_sat):
                    plt.plot(time_dictionary['sat_'+str(node)+'_x'][0], time_dictionary['sat_'+str(node)+'_y'][0], 'o', color='red')
                for node in range(n_sat):
                    plt.plot(time_dictionary['sat_'+str(node)+'_x'][ep], time_dictionary['sat_'+str(node)+'_y'][ep], 'X', color='blue')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('step: '+str(ep))
            plt.savefig(f'{name_experiment}\{ep}.png')
            plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(time_dictionary['total_area'])
    plt.xlabel('Iterations')
    plt.ylabel('Area [m^2]')
    plt.title('Total intersection area')
    plt.savefig(f'{name_experiment}\history.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(time_dictionary['energy'])
    plt.xlabel('Iterations')
    plt.ylabel('Energy')
    plt.title('Total energy system')
    plt.savefig(f'{name_experiment}\energy.png')
    plt.close()


class Constellation:
    """
    n_sat -> number of satellites in the constellation
    n_dir -> number of directions a satellite can perform
    radius -> a satellite withing the range is considered a neighbor
    min_distance -> minimum distance between satellites
    max_distance -> maximum distance between satellites
    """

    def __init__(self, n_sat=10, n_dir=3, radius=1, min_distance=0.1, max_distance=2, custom_const=False, col_lines=[]) -> None:
        self.n_sat = len(custom_const) if custom_const else n_sat
        self.n_dir = n_dir
        self.radius = radius
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.custom_const = custom_const  
        self.const_graph = nx.Graph()  
        self.qubits_graph = nx.Graph()
        self.tot_dist = 0
        self.hist = {str(i): list() for i in range(self.n_sat)}
        self.col_lines = col_lines
        # idx*self.n_dir+dir

        if self.custom_const:
            for idx, node in enumerate(self.custom_const):
                self.const_graph.add_node(str(idx),                                                             # '0'
                                          pos_x=node[0], pos_y=node[1],                                         # .., ..
                                          )
                for dir in range(self.n_dir):
                    self.qubits_graph.add_node(str(idx)+"-"+str(dir),
                                               q_binary = Binary(str(idx)+"-"+str(dir)))
        else:
            for idx in range(n_sat):
                while True:
                    pos_x = np.random.normal(scale=0.2)*self.n_sat
                    pos_y = np.random.normal(scale=0.2)*self.n_sat
                    if self._check_min_distance(pos_x, pos_y, self.min_distance):
                        self.const_graph.add_node(str(idx), 
                                                pos_x=pos_x, pos_y=pos_y,
                                                )
                        for dir in range(self.n_dir):
                            self.qubits_graph.add_node(str(idx)+"-"+str(dir),
                                                       q_binary = Binary(str(idx)+"-"+str(dir)))
                        break

        self.max_dist_dict = {str(i): 0 for i in range(self.n_sat)}
        for sat_1 in self.const_graph.nodes():
            dist_list = list()
            for sat_2 in self.const_graph.nodes():
                distance = ((self.const_graph.nodes[sat_1]['pos_x'] - self.const_graph.nodes[sat_2]['pos_x'])**2 + (self.const_graph.nodes[sat_1]['pos_y'] - self.const_graph.nodes[sat_2]['pos_y'])**2)**0.5
                # print(str(sat_1)+"-"+str(sat_2), distance, self.const_graph.has_edge(sat_1, sat_2))
                dist_list.append(distance)
                if distance <= self.radius and sat_1 != sat_2 and (not self.const_graph.has_edge(sat_1, sat_2)):
                    self.const_graph.add_edge(sat_1, sat_2, weight=distance)
            self.max_dist_dict[str(sat_1)] = max(dist_list)

        for qubit_1 in range(self.n_dir * self.n_sat):
            node_1 = str(qubit_1 // self.n_dir)
            qubit_node_1 = qubit_1 % self.n_dir
            for qubit_2 in range(qubit_1, self.n_dir * self.n_sat):
                node_2 = str(qubit_2 // self.n_dir)
                qubit_node_2 = qubit_2 % self.n_dir
                new_node_1_pos_x = self.const_graph.nodes[node_1]['pos_x'] + math.cos(qubit_node_1*2*math.pi/self.n_dir)
                new_node_1_pos_y = self.const_graph.nodes[node_1]['pos_y'] + math.sin(qubit_node_1*2*math.pi/self.n_dir)
                new_node_2_pos_x = self.const_graph.nodes[node_2]['pos_x'] + math.cos(qubit_node_2*2*math.pi/self.n_dir)
                new_node_2_pos_y = self.const_graph.nodes[node_2]['pos_y'] + math.sin(qubit_node_2*2*math.pi/self.n_dir)
                new_distance = (((new_node_1_pos_x - new_node_2_pos_x)**2 + (new_node_1_pos_y - new_node_2_pos_y)**2)**0.5 - self.max_distance)**2
                # print(node_1+"-"+str(qubit_node_1), node_2+"-"+str(qubit_node_2), new_distance, self.const_graph.has_edge(node_1, node_2))
                if qubit_1 == qubit_2:
                    pass
                elif node_1 == node_2:
                    pass
                    # self.qubits_graph.add_edge(node_1+"-"+str(qubit_node_1), node_2+"-"+str(qubit_node_2), weight=0)
                elif self.const_graph.has_edge(node_1, node_2):
                    self.qubits_graph.add_edge(node_1+"-"+str(qubit_node_1), node_2+"-"+str(qubit_node_2), weight=new_distance) 

        edge_weights = [d['weight'] for u, v, d in self.qubits_graph.edges(data=True)]

        max_weight = max(edge_weights)

        for u, v, d in self.qubits_graph.edges(data=True):
            # d['weight'] = 1/(1 + np.exp(-d['weight'] / max_weight)) 
            d['weight'] = d['weight'] / max_weight

    def get_centroid(self):
        c_x = 0
        c_y = 0
        for node in self.const_graph.nodes():
            c_x = c_x + self.const_graph.nodes[node]['pos_x']
            c_y = c_y + self.const_graph.nodes[node]['pos_y']

        c_x /= self.n_sat
        c_y /= self.n_sat

        minimum = np.inf

        for node in self.const_graph.nodes():
            pos_x = self.const_graph.nodes[node]['pos_x']
            pos_y = self.const_graph.nodes[node]['pos_y']
            distance = ((c_x - pos_x)**2 + (c_y - pos_y)**2)**0.5
            if distance < minimum:
                minimum = distance
                new_c_x = pos_x
                new_c_y = pos_y

        return new_c_x, new_c_y

    def tot_area_covered(self):
        R = self.max_distance / 2
        self.max_area = (self.max_distance/2)**2 * math.pi * self.n_sat
        self.tot_area = (self.max_distance/2)**2 * math.pi * self.n_sat
        self.total_intersection_area = 0
        for edge in self.const_graph.edges():
            d = self.const_graph.edges[edge]['weight']
            if d >= 2 * R:
                intersection_area = 0
            else:
                intersection_area = R**2 * math.acos((d**2 + R**2 - R**2) / (2 * d * R))
                intersection_area += R**2 * math.acos((d**2 + R**2 - R**2) / (2 * d * R))
                intersection_area -= 0.5 * math.sqrt((-d + R + R) * (d + R - R) * (d - R + R) * (d + R + R))

            self.total_intersection_area = self.total_intersection_area + intersection_area
            # self.tot_area = self.tot_area - intersection_area
        
        return self.total_intersection_area # self.max_area - self.tot_area
                    
    def const_graph_comp(self):

        return self.const_graph
    
    def qubits_graph_comp(self):

        return self.qubits_graph
    
    def _check_min_distance(self, pos_x, pos_y, min_distance):
        for node in self.const_graph.nodes():
            distance = ((self.const_graph.nodes[node]['pos_x'] - pos_x)**2 + (self.const_graph.nodes[node]['pos_y'] - pos_y)**2)**0.5
            if distance < min_distance:
                return False
        return True

    def draw_constellation(self):
        for node in self.const_graph.nodes():
            plt.scatter(self.const_graph.nodes[node]['pos_x'], self.const_graph.nodes[node]['pos_y'], color='skyblue', s=200)
            plt.text(self.const_graph.nodes[node]['pos_x'], self.const_graph.nodes[node]['pos_y'], node, ha='center', va='center', color='black', fontsize=10)

        seen = []
        for edge in self.const_graph.edges():
            node_1 = str(edge[0])
            node_2 = str(edge[1])
            if ((node_1, node_2) or (node_2, node_1)) not in seen:
                seen.append((node_1, node_2))
                seen.append((node_2, node_1))
                plt.plot([self.const_graph.nodes[node_1]['pos_x'], self.const_graph.nodes[node_2]['pos_x']], [self.const_graph.nodes[node_1]['pos_y'], self.const_graph.nodes[node_2]['pos_y']], color='black', linewidth=0.1)
                # plt.text((self.const_graph.nodes[node_1]['pos_x']+self.const_graph.nodes[node_2]['pos_x'])/2, (self.const_graph.nodes[node_1]['pos_y']+self.const_graph.nodes[node_2]['pos_y'])/2, str(round(self.const_graph.edges[edge]['weight'],2)), ha='center', va='center', color='black', fontsize=10)

        plt.show()

    def draw_qubit_constellation(self):
        for node in self.qubits_graph.nodes():
            plt.scatter(self.qubits_graph.nodes[node]['pos_x'], self.qubits_graph.nodes[node]['pos_y'], color='skyblue', s=200)
            plt.text(self.qubits_graph.nodes[node]['pos_x'], self.qubits_graph.nodes[node]['pos_y'], node, ha='center', va='center', color='black', fontsize=10)

        seen = []
        for edge in self.qubits_graph.edges():
            node_1 = str(edge[0])
            node_2 = str(edge[1])
            if ((node_1, node_2) or (node_2, node_1)) not in seen:
                seen.append((node_1, node_2))
                seen.append((node_2, node_1))
                plt.plot([self.qubits_graph.nodes[node_1]['pos_x'], self.qubits_graph.nodes[node_2]['pos_x']], [self.qubits_graph.nodes[node_1]['pos_y'], self.qubits_graph.nodes[node_2]['pos_y']], color='black', linewidth=0.1)

        plt.show()

    def fake_bitstring_gen(self):
        return [rd.choice([0, 1]) for _ in range(self.n_sat * self.n_dir)]
    
    def step(self, bitstring):
        qubits_dict = dict()
        for node_qubit, value in bitstring.items():
            if node_qubit[0] != 's':
                prefisso = node_qubit.split('-')[0]
                if prefisso not in qubits_dict:
                    qubits_dict[prefisso] = [value]
                else:
                    qubits_dict[prefisso].append(value)

        new_qubits_dict = dict()
        for node, qubits in qubits_dict.items():
            if sum(qubits) > 0:
                new_qubits_dict[node] = qubits

        for node, qubits in qubits_dict.items():
            agg_x = 0
            agg_y = 0
            for idx, qubit in enumerate(qubits):
                agg_x = agg_x + qubit*math.cos(idx*2*math.pi/self.n_dir)
                agg_y = agg_y + qubit*math.sin(idx*2*math.pi/self.n_dir)

            new_pos_x = self.const_graph.nodes[node]['pos_x'] + agg_x/((agg_x**2+agg_y**2)**0.5) if sum(qubits) != 0 else self.const_graph.nodes[node]['pos_x']
            new_pos_y = self.const_graph.nodes[node]['pos_y'] + agg_y/((agg_x**2+agg_y**2)**0.5) if sum(qubits) != 0 else self.const_graph.nodes[node]['pos_y']
            self.hist[node].append((self.const_graph.nodes[node]['pos_x'], self.const_graph.nodes[node]['pos_y']))
            self.const_graph.nodes[node]['pos_x'] = new_pos_x
            self.const_graph.nodes[node]['pos_y'] = new_pos_y

        for node_1 in self.const_graph.nodes():
            for node_2 in self.const_graph.nodes():
                distance = ((self.const_graph.nodes[node_1]['pos_x'] - self.const_graph.nodes[node_2]['pos_x'])**2 + (self.const_graph.nodes[node_1]['pos_y'] - self.const_graph.nodes[node_2]['pos_y'])**2)**0.5
                if node_1 != node_2:
                    if distance <= self.radius and self.const_graph.has_edge(node_1, node_2):
                        self.const_graph.edges[(node_1, node_2)]['weight'] = distance
                    # elif distance > self.radius and self.const_graph.has_edge(node_1, node_2):
                    #     self.const_graph.remove_edge(node_1, node_2)
                    # elif distance <= self.radius and (not self.const_graph.has_edge(node_1, node_2)):
                    #     self.const_graph.add_edge(node_1, node_2, weight=distance)

        for qubit_1 in range(self.n_dir * self.n_sat):
            node_1 = str(qubit_1 // self.n_dir)
            qubit_node_1 = qubit_1 % self.n_dir
            for qubit_2 in range(qubit_1, self.n_dir * self.n_sat):
                node_2 = str(qubit_2 // self.n_dir)
                qubit_node_2 = qubit_2 % self.n_dir
                new_node_1_pos_x = self.const_graph.nodes[node_1]['pos_x'] + math.cos(qubit_node_1*2*math.pi/self.n_dir)
                new_node_1_pos_y = self.const_graph.nodes[node_1]['pos_y'] + math.sin(qubit_node_1*2*math.pi/self.n_dir)
                new_node_2_pos_x = self.const_graph.nodes[node_2]['pos_x'] + math.cos(qubit_node_2*2*math.pi/self.n_dir)
                new_node_2_pos_y = self.const_graph.nodes[node_2]['pos_y'] + math.sin(qubit_node_2*2*math.pi/self.n_dir)
                new_distance = ((new_node_1_pos_x - new_node_2_pos_x)**2 + (new_node_1_pos_y - new_node_2_pos_y)**2)**0.5
                # print(node_1+"-"+str(qubit_node_1), node_2+"-"+str(qubit_node_2), new_distance, self.const_graph.has_edge(node_1, node_2))
                if qubit_1 != qubit_2:
                    if node_1 == node_2:
                        pass
                        # self.qubits_graph.add_edge(node_1+"-"+str(qubit_node_1), node_2+"-"+str(qubit_node_2), weight=0)
                    elif self.const_graph.has_edge(node_1, node_2) and self.qubits_graph.has_edge(node_1+"-"+str(qubit_node_1), node_2+"-"+str(qubit_node_2)):
                        if new_distance >= max_distance:
                            new_distance = (new_distance - self.max_distance)**2
                        elif 0 < new_distance < max_distance:
                            new_distance = (new_distance - self.max_distance)**4
                        else: 
                            new_distance = math.inf
                        self.qubits_graph.edges[(node_1+"-"+str(qubit_node_1), node_2+"-"+str(qubit_node_2))]['weight'] = new_distance
                    # elif self.const_graph.has_edge(node_1, node_2) and (not self.qubits_graph.has_edge(node_1+"-"+str(qubit_node_1), node_2+"-"+str(qubit_node_2))):
                    #     self.qubits_graph.add_edge(node_1+"-"+str(qubit_node_1), node_2+"-"+str(qubit_node_2), weight=new_distance)
                    # elif (not self.const_graph.has_edge(node_1, node_2)) and self.qubits_graph.has_edge(node_1+"-"+str(qubit_node_1), node_2+"-"+str(qubit_node_2)):
                    #     self.qubits_graph.remove_edge(node_1+"-"+str(qubit_node_1), node_2+"-"+str(qubit_node_2))
                        
        edge_weights = [d['weight'] for u, v, d in self.qubits_graph.edges(data=True)]

        max_weight = max(edge_weights)

        for u, v, d in self.qubits_graph.edges(data=True):
            # d['weight'] = 1/(1 + np.exp(-d['weight'] / max_weight))  
            d['weight'] = d['weight'] / max_weight


def get_gauss_vale_dict(c_x, c_y, n_sat, max_distance):
    const_graph = const.const_graph_comp()
    qubits_graph = const.qubits_graph_comp()
    gauss_dict = {str(i): 0 for i in range(n_sat)}
    for idx, node in enumerate(const_graph.nodes()):
        gauss_dict[str(idx)] = bidim_gauss(x=const_graph.nodes[node[0]]['pos_x']/n_sat,
                                           y=const_graph.nodes[node[0]]['pos_y']/n_sat,
                                           c_x=c_x,
                                           c_y=c_y,
                                           sigma_x=1,
                                           sigma_y=1,
                                           A=1) # n_sat)
        
    return gauss_dict


def model_creation(mode, const, n_dir, constraints, gauss_vale_dict):
    const_graph = const.const_graph_comp()
    qubits_graph = const.qubits_graph_comp()
    
    bqm = BinaryQuadraticModel("BINARY")
    terms_inside = list()
    for edge in qubits_graph.edges():
        qubit_1, qubit_2 = edge[0], edge[1]
        weight = qubits_graph.edges[edge]['weight']
        bqm.add_variable(qubit_1)
        bqm.add_variable(qubit_2)
        bqm.add_interaction(qubit_1, qubit_2, weight)   # da aggiungere lo "strength"
        
    if constraints:
        for idx, node in enumerate(const_graph.nodes()):
            terms = list()
            for dir in range(n_dir):
                terms.append((node+"-"+str(dir), 1))
                # bqm.add_linear((node+"-"+str(dir)), gauss_vale_dict[str(idx)])

            bqm.add_linear_inequality_constraint(terms=terms, lagrange_multiplier=1, lb=0, ub=(n_dir-1), label=str(idx))

    return bqm 


if __name__ == '__main__':

    n_sat = 15                       # how many satellites
    radius = 10000                     # Within this radius, a satellite is considered a neighboring
    n_dir = 3                        # how many dicrections a satellite can perform
    min_distance = math.sqrt(n_sat)               # minimum initial distance between two satellites
    max_distance = n_sat*3.5                # maximum distance in order to have a connection
    epochs = 1500
    constraints = True
    custom_const = [[0,0],  [0,25],  [0,50],
                    [25,0], [25,25], [25,50], 
                    [50,0], [50,25], [50,50]]
    mode = 'Tabu'                   # Tabu, Exact, Tree, Fake, Steepest
    
    col_lines = generate_colors(n_sat)

    const = Constellation(n_sat, n_dir, radius, min_distance, max_distance, col_lines=col_lines)
    # const = Constellation(n_sat, n_dir, radius, min_distance, max_distance, custom_const, col_lines=col_lines)

    c_x, c_y = const.get_centroid()
    gauss_vale_dict = get_gauss_vale_dict(c_x, c_y, n_sat, max_distance)

    while True:
        random_number = np.random.random_integers(low=0, high=1000)
        name_experiment = str(random_number)+"_"+mode+"_sat_"+str(n_sat)+"_dir_"+str(n_dir)+"_max_dist_"+str(max_distance)
        if not os.path.exists(name_experiment):
            os.mkdir(name_experiment)
            break 

    print("Folder number: ", random_number)

    time_process = 0
    time_dictionary = {'it': [], 'it_time': [], 'total_time': [], 'total_area': [], 'energy': []}

    for i in range(n_sat):
        time_dictionary['sat_'+str(i)+'_x'] = []
        time_dictionary['sat_'+str(i)+'_y'] = []

    for stp in range(epochs):

        # const.draw_constellation()
        # const.draw_qubit_constellation()
        # for edge in const.qubits_graph.edges(): 
        #     print(edge, const.qubits_graph.edges[edge]['weight'])

        model = model_creation(mode, const, n_dir, constraints, gauss_vale_dict)

        if mode == 'Tabu':
            sampler = TabuSampler()
            t1 = time.time()
            spl = sampler.sample(model, tenure=1, num_restarts=1)
            t2 = time.time()
            bitstring = spl.first.sample
            energy = spl.first.energy
            delta_time = t2 - t1

        elif mode == 'DWave':
            sampler = EmbeddingComposite(DWaveSampler(token=DWAVE_API_TOKEN))
            spl = sampler.sample(model, num_reads=1)
            bitstring = spl.first.sample
            energy = spl.first.energy
            delta_time = spl.info['timing']['qpu_access_time']

        elif mode == 'Steepest':
            sampler = SteepestDescentSolver()
            t1 = time.time()
            spl = sampler.sample(model)
            t2 = time.time()
            bitstring = spl.first.sample
            energy = spl.first.energy
            delta_time = t2 - t1

        elif mode == 'Simulated':
            sampler = SimulatedAnnealingSampler()
            t1 = time.time()
            spl = sampler.sample(model)
            t2 = time.time()
            bitstring = spl.first.sample
            energy = spl.first.energy
            delta_time = t2 - t1

        elif mode == 'Fake':
            bitstring = const.fake_bitstring_gen()

        time_process = time_process + delta_time
        tot_a = const.tot_area_covered()
        print(stp, tot_a, delta_time)

        time_dictionary['it'].append(stp)
        time_dictionary['it_time'].append(delta_time)
        time_dictionary['total_time'].append(time_process)
        time_dictionary['total_area'].append(tot_a)
        time_dictionary['energy'].append(energy)
        
        for node in range(n_sat):
            time_dictionary['sat_'+str(node)+'_x'].append(const.const_graph.nodes[str(node)]['pos_x'])
            time_dictionary['sat_'+str(node)+'_y'].append(const.const_graph.nodes[str(node)]['pos_y'])

        const.step(bitstring)

        # plt.savefig(f'{name_experiment}\{stp}.png')
        # plt.close()
        pd.DataFrame(time_dictionary).to_csv(f'{name_experiment}/data.csv')


plot_trajectories(epochs, n_sat, time_dictionary, name_experiment)
 
    # implementare la  {q_1: {q_2: distanza, q_3: distanza, q_4: distanza}} hash table per l'aggiornamento dei pesi (dizionario)
    # provare con 1000 satelliti quanto ci mette a risolvere il solver il problema di ottimizzazione
    # pascal (da tenere d'occhio perchè crea i processori quantistici)