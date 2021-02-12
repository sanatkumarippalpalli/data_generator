import json
import numpy as np
import pandas as pd
import random
from datetime import datetime
from robustcontrol.utils import InternalDelay, tf, mimotf
from scipy.signal import detrend

def GBN_seq(N, p_swd, Nmin=1, Range=[-1.0, 1.0], Tol = 0.01, nit_max = 30):
    min_Range = min(Range)
    max_Range = max(Range)
    prob = np.random.random()
    # set first value
    if prob < 0.5:
        gbn = -1.0*np.ones(N)
    else:
        gbn = 1.0*np.ones(N)
    # init. variables
    p_sw = p_sw_b = 2.0             # actual switch probability
    nit = 0; 
    while (np.abs(p_sw - p_swd))/p_swd > Tol and nit <= nit_max:
        i_fl = 0; Nsw = 0;
        for i in range(N - 1):
            gbn[i + 1] = gbn[i]
            # test switch probability
            if (i - i_fl >= Nmin):
                prob = np.random.random()
                # track last test of p_sw
                i_fl = i
                if (prob < p_swd):
                    # switch and then count it
                    gbn[i + 1] = -gbn[i + 1]
                    Nsw = Nsw + 1
        # check actual switch probability
        p_sw = Nmin*(Nsw+1)/N; #print("p_sw", p_sw);
        # set best iteration
        if np.abs(p_sw - p_swd) < np.abs(p_sw_b - p_swd):
            p_sw_b = p_sw
            Nswb = Nsw
            gbn_b = gbn.copy()
        # increase iteration number
        nit = nit + 1; #print("nit", nit)
    # rescale GBN
    for i in range(N):
        if gbn_b[i] > 0.:
            gbn_b[i] = max_Range
        else:
            gbn_b[i] = min_Range
    return gbn_b, p_sw_b, Nswb
input_file = 'model.json'
with open(input_file, 'r') as f:
    input_data = json.load(f)

outputs = list(input_data['models'].keys())
inputs = list(input_data['models'][outputs[0]].keys())

data = pd.DataFrame()
N = input_data['data_length']
input_switch_probability = list(input_data['input_switch_probability'].values())
block_sizes = list(input_data['input_block_size'].values())
input_ranges = list(input_data['range'].values())
process_tss = input_data['time_to_steady_state']
for idx, input in enumerate(inputs):
    random.seed(datetime.now())
    p_swd = random.sample(range(input_switch_probability[idx][0],input_switch_probability[idx][1]),1)[0] / 100
    s = pd.Series()
    block_size = block_sizes[idx]
    input_range = input_ranges[idx]
    while len(s) <= N:
        a, b = random.sample(range(input_range[0], input_range[1]), 2)
        Nmin = random.sample(range(process_tss//5,process_tss),1)[0]
        i_range = list()
        i_range.append(min(a, b))
        i_range.append(max(a, b))
        U, _, _ = GBN_seq(block_size, p_swd, Nmin, Range=i_range, Tol = 0.01, nit_max = 30)
        s = s.append(pd.Series(U), ignore_index=True)
    data[input] = s[:N]


G = list()
for output in outputs:
    G_i = list()
    for input in inputs:
        if input_data['models'][output][input]:
            num = input_data['models'][output][input][0]
            den = input_data['models'][output][input][1]
            theta = input_data['models'][output][input][2]
            G_ij = tf(num, den,deadtime= theta)
        else:
            G_ij = tf([0], [1])
        G_i.append(G_ij)
    G.append(G_i)


G = mimotf(G)
G_id = InternalDelay(G)
input = data[inputs].to_numpy()
uf = lambda t: input[int(t)]
ts = np.arange(0, N-1, 1)
ys = G_id.simulate(uf, ts)

yout_df = pd.DataFrame(ys, columns=outputs)
data = pd.concat([data, yout_df], axis=1)
start = input_data['start_date']
freq = input_data['sampling']
data['Time'] = pd.date_range(start=start, periods=len(data) , freq=freq)
data.set_index('Time', inplace=True)
data.dropna(axis=0, inplace=True)
is_noise = input_data['add_noise']
noise_std = list(input_data['output_noise_std'].values())
for idx, output in enumerate(outputs):
    detrended = detrend(data[output])
    if is_noise:
        data[output] = detrended + abs(min(detrended)) + np.random.normal(0,noise_std[idx],len(detrended))
    else:
        data[output] = detrended + abs(min(detrended))

out_file = input_data['outputfilename']
data.iloc[21:].to_csv(out_file)