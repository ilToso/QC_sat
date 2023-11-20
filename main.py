
import copy
import math
import random as rd
import matplotlib.pyplot as plt

class Satellite:
    """
    pos_x (float) -> Satellite's x position
    pos_y (float) -> Satellite's y position
    neighbors_distance (dict) -> satellite index (str) : Euclidian distance (flaot)
    qubits (list) -> list of int from 0 to # of directions
    """

    def __init__(self, pos_x = 0, pos_y = 0, qubits=[]) -> None:
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.neighbors_distance = {} 
        self.neighbors_qubit_distance = {}        
        self.qubits = qubits

    def get_pos_x(self):
        return self.pos_x
    
    def get_pos_y(self):
        return self.pos_y

    def move(self, actions: list, n_dir):
        for i in range(n_dir):
            self.pos_x = self.pos_x + actions[i]*math.cos(i*2*math.pi/n_dir)
            self.pos_y = self.pos_y + actions[i]*math.sin(i*2*math.pi/n_dir)

    # def add_neighbor_distance(self, neigh_sat, dist):
    #     self.neighbors_distance[neigh_sat] = dist

    # def get_neighbor_distance(self):
    #     return self.neighbors_distance

    # def add_neighbor_qubit_distance(self, neigh_qubit, dist):
    #     self.neighbors_qubit_distance[neigh_qubit] = dist

    # def get_neighbor_qubit_distance(self):
    #     return self.neighbors_qubit_distance
    
    def get_qubits(self):
        return self.qubits

        
class Constellation:
    """
    n_sat -> number of satellites in the constellation
    n_dir -> number of directions a satellite can perform
    radius -> a satellite withing the range is considered a neighbor
    min_distance -> minimum distance between satellites
    max_distance -> maximum distance between satellites
    """

    def __init__(self, n_sat=10, n_dir=3, radius=1, min_distance=0.1, max_distance=2) -> None:
        self.n_sat = n_sat
        self.n_dir = n_dir
        self.constellation = {}             # constellation['0'] = Satellite()
        self.satellite_distances = {}
        self.couplers_constellation = {}

        # Aggiungi gli altri satelliti garantendo una distanza minima
        for i in range(n_sat):
            while True:
                new_satellite = Satellite(rd.random(), rd.random(), qubits=[j for j in range(n_dir)])
                if self._check_min_distance(new_satellite, min_distance):
                    self.constellation[str(i)] = new_satellite
                    break

        for sat_idx_1, sat_1 in self.constellation.items():
            for sat_idx_2, sat_2 in self.constellation.items():
                sat_str_1_2 = sat_idx_1 + '-' + sat_idx_2
                sat_str_2_1 = sat_idx_2 + '-' + sat_idx_1
                distance = ((sat_1.get_pos_x() - sat_2.get_pos_x())**2 + (sat_1.get_pos_y() - sat_2.get_pos_y())**2)**0.5
                if distance <= radius and sat_1 != sat_2 and (not self.satellite_distances.has_key(sat_str_1_2)) and (not self.satellite_distances.has_key(sat_str_2_1)):
                    self.satellite_distances[sat_str_1_2] = distance
                    self.satellite_distances[sat_str_2_1] = distance

        for qubit_1 in range(self.n_dir * self.n_sat):
            for qubit_2 in range(qubit_1, self.n_dir * self.n_sat):
                qubit_couple_str_1_2 = str(qubit_1) + '-' + str(qubit_2)
                qubit_couple_str_2_1 = str(qubit_2) + '-' + str(qubit_1)
                if qubit_1 == qubit_2:
                    self.couplers_constellation[qubit_couple_str_1_2] = 0.1
                elif qubit_2 < qubit_1 + self.n_dir:
                    self.couplers_constellation[qubit_couple_str_1_2] = 0.2
                    self.couplers_constellation[qubit_couple_str_2_1] = 0.2
                else:
                    sat_1 = str(qubit_1 // n_dir)
                    sat_2 = str(qubit_2 // n_dir)
                    sat_str = sat_1 + '-' + sat_2
                    qubit_sat_1 = qubit_1 % n_dir
                    qubit_sat_2 = qubit_2 % n_dir
                    distance = self.satellite_distances[sat_str]
                    new_distance = distance + (math.cos(qubit_sat_1*2*math.pi/n_dir)**2 + math.cos(qubit_sat_2*2*math.pi/n_dir)**2)**0.5
                    self.couplers_constellation[qubit_couple_str_1_2] = new_distance
                    self.couplers_constellation[qubit_couple_str_2_1] = new_distance 

                    
        for sat_idx_1, sat_1 in self.constellation.items():
            for sat_idx_2, sat_2 in self.constellation.items():
                for q_sat_1 in sat_1.get_qubits():
                    for q_sat_2 in sat_2.get_qubits():
                        if sat_1 != sat_2 and q_sat_1 != q_sat_2:
                            future_sat_1_pos_x = sat_1.get_pos_x() + math.cos(q_sat_1*2*math.pi/n_dir)
                            future_sat_1_pos_y = sat_1.get_pos_y() + math.sin(q_sat_1*2*math.pi/n_dir)
                            future_sat_2_pos_x = sat_2.get_pos_x() + math.cos(q_sat_2*2*math.pi/n_dir)
                            future_sat_2_pos_y = sat_2.get_pos_y() + math.sin(q_sat_2*2*math.pi/n_dir)
                            future_distance = (((future_sat_1_pos_x - future_sat_2_pos_x)**2 + (future_sat_1_pos_y - future_sat_2_pos_y)**2)**0.5 - max_distance)**2
                            sat_1.add_neighbor_qubit_distance(sat_idx_1 + '/' + q_sat_1 + '-' + sat_idx_2 + '/' + q_sat_2, future_distance)
                            sat_2.add_neighbor_qubit_distance(sat_idx_2 + '/' + q_sat_2 + '-' + sat_idx_1 + '/' + q_sat_1, future_distance)

    def _check_min_distance(self, new_satellite, min_distance):
        for idx, sat in self.constellation.items():
            distance = ((sat.get_pos_x() - new_satellite.get_pos_x())**2 + (sat.get_pos_y() - new_satellite.get_pos_y())**2)**0.5
            if distance < min_distance:
                return False
        return True

    def draw_constellation(self):
        for idx, sat in self.constellation.items():
            plt.scatter(sat.get_pos_x(), sat.get_pos_y(), color='skyblue', s=100)
            plt.text(sat.get_pos_x(), sat.get_pos_y(), idx, ha='center', va='center', color='black', fontsize=8)

        for idx, sat in self.constellation.items():
            couples = sat.get_neighbor_distance().keys()
            seen = []
            for cpl in couples:
                sat_1, sat_2 = cpl.split("-")
                if ((sat_1, sat_2) or (sat_2, sat_1)) not in seen:
                    seen.append((sat_1, sat_2))
                    seen.append((sat_2, sat_1))
                    plt.plot([self.constellation[sat_1].get_pos_x(), self.constellation[sat_2].get_pos_x()], [self.constellation[sat_1].get_pos_y(), self.constellation[sat_2].get_pos_y()], color='black', linewidth=0.5)

        plt.show() 

    #def Q_matrix_comp(self):



if __name__ == '__main__':

    n_sat = 40                      # how many satellites
    radius = 0.                    # Within this radius, a satellite is considered a neighboring
    n_dir = 3                       # how many dicrections a satellite can perform
    min_distance = 0.1              # minimum initial distance between two satellites
    max_distance = 2                # maximum distance in order to have a connection

    const = Constellation(n_sat, n_dir, radius, min_distance, max_distance)

    const.draw_constellation()

    a = 1

    