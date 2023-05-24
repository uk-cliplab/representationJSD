import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import repitl.kernel_utils as ku
from TwoSampleTest import run_experiment_Higgs
from utils import sample_blobs, HDGM
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--experiment-name', required=True, help='Experiment name for saved files', type=str)
parser.add_argument('-nPerm', '--permTestSize', required= True, help='Number of permutations to compute power of test',type=int )
parser.add_argument('-nTest', '--numTestSets', required= True, help='Number of test sets to evaluate the learned parameters', type=int)
parser.add_argument('-sign', '--significance', required= True, help='significance level to reject the null hypothesis', type=float )
parser.add_argument('-rId', '--repId',required=True, help = 'repetition Id to store the results', type=int)
parser.add_argument('-datafolder', '--DATAFOLDER', required = True, type = str  )

args = parser.parse_args()
# parameter of the 2ST
n_samples = [500,1000,1500,2500,4000,5000]
numRepetitions = 10 # each core will do one repetition
# permTestSize = 100
# numTestSets = 10
# significance = 0.05

def run():
    dummy = np.zeros(10)
	
    fname = args.DATAFOLDER + '/results_higgs_' + str(args.repId) + '.npz'
 
    np.savez(fname, dummy)
    jsd_results_rff, jsd_results_ff, deep_jsd_results, mmd_results, deep_mmd_results, c2st_s_results, c2st_l_results = run_experiment_Higgs(n_samples, numRepetitions,args.numTestSets, args.permTestSize, args.significance, parallel=True, pId= args.repId)
    results_higgs = [jsd_results_rff, jsd_results_ff, deep_jsd_results, mmd_results, deep_mmd_results, c2st_s_results, c2st_l_results]
    np.savez(fname, *results_higgs)

if __name__ == "__main__":
    run()
