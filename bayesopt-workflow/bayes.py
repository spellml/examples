"""
Bayesian Optimization workflow using Spell API
"""
import spell.client
import argparse
import json
import asyncio
import subprocess
import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from functools import reduce
from threading import Thread, Lock

client = spell.client.from_environment()
utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
lock = Lock()

parser = argparse.ArgumentParser()

parser.add_argument('--params', type=json.loads,
                    dest='params', help='hyperparameter domain dictionary mapping hyperparameter name to a tuple of (min, max, type)',
                    default={'batch-size': (32, 256, 'int'), 'dropout': (.1, .3, 'float')})
parser.add_argument('--command', type=str,
                    dest='command', help='the spell run command to run',
                    default="python cifar.py")
parser.add_argument('--metric', type=str,
                    dest='metric', help='the metric by which your model will be evaluated',
                    default="keras/val_acc")
parser.add_argument('--parallel-runs', type=int,
                    dest='parallel', help='number of runs that will be spawned in parallel',
                    default=3)
parser.add_argument('--iterations', type=int,
                    dest='iterations', help='number of times new parallel runs will be spawned',
                    default=3)
parser.add_argument('--optimize', type=str,
                    dest='optimize', help='maximize or minimize', default="maximize")
parser.add_argument('--machine-type', type=str,
                    dest='machine', help='machine type', default="K80")
parser.add_argument('--metric-type', type=str,
                    dest='type', help='How to evaluate metric (average, last, max)', default="average")
args = parser.parse_args()


class SER:
    """
    Represents the suggest-evaluate-register bayesian optimization
    paradigm corresponding to a last suggested param (last_param)
    and a corresponding SER output (last_output). Maintains invariant that
    an instance of SER will not "suggest" a new point before having
    registered a new evaluated datapoint.
    """
    def __init__(self):
        self.last_param = None
        self.last_output = None

    def step(self, optimizer, f):
        # need to lock so that no other thread can interleave and register a point
        # before the current thread finishes 1. registering its last point and 2. calling
        # suggest to get its next point
        lock.acquire()
        try:
            if self.last_output is not None:
                # print('registering {0} --> {1}'.format(self.last_param, self.last_output))
                optimizer.register(self.last_param, self.last_output)
            self.last_param = optimizer.suggest(utility)
        finally:
            lock.release()
            self.last_output = f(**self.last_param)

    def finish(self, optimizer):
        optimizer.register(self.last_param, self.last_output)

def black_box_function(**pvals):
    """
    black_box_function required by Bayesian Optimizer

    @param pvals, a flexible length dictionary mapping hyperparameters
    to chosen values
    @throws Exception if metric to track is not found in metrics for the run
    """

    # set up python command to run model
    command = args.command
    for k,v in pvals.items():
        if args.params[k][2] == 'int':
            v = int(v)
        command += ' --{0} {1}'.format(k, v)

    # spawn a spell run using Spell API
    r = client.runs.new(
        machine_type=args.machine,
        command=command,
        pip_packages=["idx2numpy", "argparse",],
        framework="tensorflow",
        commit_label="label",
    )

    print("\t[{0}] run spawned: $ {1}".format(r.id, command))

    # get metric value
    metrics = []
    for m in r.metrics(metric_name=args.metric, follow=True):
        metrics.append(m[2])
    if not metrics:
        raise Exception('Run ended prematurely or metric ({0}) was not found (check run logs)'.format(args.metric))

    if args.type == 'average':
        metric_value = reduce(lambda x, y: x + y, metrics) / len(metrics)
    elif args.type == 'last':
        metric_value = metrics[-1]
    elif args.type == 'max':
        metric_value = max(metrics)
    else:
        #TODO validate at args parsing level
        raise Exception('Invalid metric type ({0})'.format(args.type))

    r.wait_status(client.runs.COMPLETE)

    print("\t[{0}] run output: {1} ({2})".format(r.id, round(metric_value, 2), args.metric))

    return metric_value if args.optimize == 'maximize' else (-1)*metric_value

def f2(x):
    """
    Testing function
    """
    x[0] = x[0]*15
    x[1] = (x[1]*15)-5

    y = np.square(x[1] - (5.1/(4*np.square(math.pi)))*np.square(x[0]) + (5/math.pi)*x[0] - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x[0]) + 10;

    result = y
    return result

def parse_params():
    """
    Parse hyperparameter input domains
    """
    tup = args.params
    param_to_min_max = {k: (v[0], v[1]) for k, v in tup.items()}
    return param_to_min_max

#set up optimizer
pbounds = parse_params()
# print('pbounds are -> {0}'.format(pbounds))
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2,
    random_state=1
)

#set up args.parallel SER runs
parallel_runs = []
for i in range(args.parallel):
    parallel_runs.append(SER())

for i in range(args.iterations):
    print('start (iteration {0})'.format(i))

    parallel_threads = list(map(lambda run: Thread(target=run.step, args=(optimizer, black_box_function)), parallel_runs))

    for thread in parallel_threads:
        thread.start()

    for thread in parallel_threads:
        thread.join()

    print('end (iteration {0})\n'.format(i))

for run in parallel_runs:
    run.finish(optimizer)

print("LOCAL MAX:")
print('\t' + str(optimizer.max))

# for i, res in enumerate(optimizer.res):
#     print("Iteration {}: \n\t{}".format(i, res))
