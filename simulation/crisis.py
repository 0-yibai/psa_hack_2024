import numpy as np
import random

TOTAL_DAYS = 1000
TOTAL_PORTS = 50

INITIAL_SHIPS = 3000
MIN_CAPACITY = 80
MAX_CAPACITY = 200
MAX_CAPACITY_ERROR = 10
MIN_DIST = 5
MAX_DIST = 20
MAX_CLOSURE = 10

C1 = 1.1

capacity = random.sample(range(MIN_CAPACITY, MAX_CAPACITY), TOTAL_PORTS)

class Utils():
    @staticmethod
    def fluctuate_but_remain_row_stochastic(arr):
        for p in arr:
            p += random.randint(-5, 5) / 100
            if p < 0:
                p = 0.1
            if p > 1:
                p = 0.9
        arr = arr / np.sum(arr)
        return arr


def get_travel_days(dist, src, dest):
    '''
    Generate days required for a ship to travel from one port
    to another, which is slightly deviated from the mean.

    @param
    dist: 2D matrix specifying distance between any 2 ports
    src: source port index
    dest: destination port index
    '''
    mean = int(dist[src][dest])
    return random.randint(mean - mean // 3, mean + mean // 3)

def get_ships_to_destination_ports(corr, src, ship_count):
    '''
    Get the number of ships to each distination port given the source port,
    calculated based on the correlation matrix, with some deviations
    '''
    ships_to_destinations = Utils.fluctuate_but_remain_row_stochastic(corr[src]) * ship_count
    return ships_to_destinations

def get_closure_period():
    return 

def port_close(data, dist, corr, day, port):
    closure_period = random.randint(1, 10)
    intake = capacity[port] + random.randint(-MAX_CAPACITY_ERROR, MAX_CAPACITY_ERROR)
    for i in range(closure_period):
        data[day+i] += intake

        # simulate aftermath
        src = port
        ships_to_destinations = get_ships_to_destination_ports(corr, src, intake).astype(int)
        for dest in range(TOTAL_PORTS):
            travel_days = get_travel_days(dist, src, dest)
            arrival_day = day + travel_days
            if arrival_day < TOTAL_DAYS:
                data[arrival_day][dest] += ships_to_destinations[dest]

