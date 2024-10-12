import copy
import numpy as np
import random
from crisis import port_close


def get_exports(length, mean):
    return np.random.chisquare(mean, length)

def get_shipping_days(dist, i, j):
    '''
    Generate days required to ship a product from one port to another, 
    which is slightly deviated from the mean.

    @param
    dist: 2D matrix specifying distance between any 2 ports
    i: row index / i-th port
    j: column index / j-th port
    '''
    mean = int(dist[i][j])
    sign = 1 if (random.randint(0, 1) == 1) else -1
    deviation = random.randint(0, mean // 3)
    return mean + sign * deviation

def get_destination_port(corr, src):
    '''
    Get the distination port given the starting port based on the correlation
    matrix.
    '''
    symbols = np.arange(TOTAL_PORTS)
    return np.random.choice(symbols, p=corr[src])

def ship(exports, data, dist, corr, i, j):
    '''
    Ship the generated export to a few randomly chosen destinations after
    randomly chosen days. These will contribute to the waiting time data.
    The ships may pass through transshipping hubs.

    @param
    exports: 2D matrix containing the amount of domestic goods to be exported
    received by each port on each day.
    i: row index / i-th day
    j: column index / j-th port / source port index
    '''
    dest_count = random.randint(1, 3)
    for i in range(dest_count):
        dest = get_destination_port(corr, j)
        shipping_days = get_shipping_days(dist, i, j)
        receival_date = i + shipping_days
        if receival_date < TOTAL_DAYS:
            data[receival_date][dest] += exports[i][j] // dest_count
            exports[receival_date][dest] += exports[i][j] // dest_count


TOTAL_DAYS = 1000
TOTAL_PORTS = 10

MIN_EXPORT = 100
MAX_EXPORT = 600

# Define distribution of export: the amount of products sent for export
# at each port on each day.
daily_export_mean = random.sample(range(MIN_EXPORT, MAX_EXPORT), TOTAL_PORTS)

# Generate exports matrix from all ports on each day.
exports = np.zeros((TOTAL_DAYS, TOTAL_PORTS))
for j in range(TOTAL_PORTS):
    exports[:, j] = get_exports(TOTAL_DAYS, daily_export_mean[j])

data = copy.deepcopy(exports)

# Generate distance between any 2 ports
dist = np.zeros((TOTAL_PORTS, TOTAL_PORTS))
for i in range(TOTAL_PORTS):
    for j in range(TOTAL_PORTS):
        dist[i][j] = random.randint(5, 20)

# Generate the correlation matrix, representing the likelihood of an item
# getting shipped from one port to another.
# Row: src
# Col: dest
corr = np.random.rand(TOTAL_PORTS, TOTAL_PORTS)
corr = corr / corr.sum(axis=1, keepdims=True)

for i in range(TOTAL_DAYS):
    for j in range(TOTAL_PORTS):
        ship(exports, data, dist, corr, i, j)

for i in range(TOTAL_DAYS):
    for j in range(TOTAL_PORTS):
        data[i][j] = int(data[i][j])

# Simulate rise in capacity
for i in range(TOTAL_DAYS // 2, TOTAL_DAYS):
    j = 5
    data[i][j] -= 100
    if data[i][j] < 0:
        data[i][j] = 0

port_close(data, dist, corr, 170, 3)
port_close(data, dist, corr, 300, 7)
port_close(data, dist, corr, 800, 1)
port_close(data, dist, corr, 830, 2)



np.savetxt("data.csv", data, delimiter=',')