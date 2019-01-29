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
parser.add_argument('--max-runs', type=int,
                    dest='maxruns', help='number of times new parallel runs will be spawned',
                    default=9)
parser.add_argument('--optimize', type=str,
                    dest='optimize', help='maximize or minimize', default="maximize", choices=['maximize', 'minimize'])
parser.add_argument('--machine-type', type=str,
                    dest='machine', help='machine type', default="K80", choices=['K80', 'CPU'])
parser.add_argument('--metric-type', type=str,
                    dest='type', help='How to evaluate metric (average, last, max)', default="average", choices=['average', 'last', 'max'])
args = parser.parse_args()


class ParallelRun:
    """
    Represents a parallel run with a last evaluated param (last_param),
    and a corresponding output (last_output). Ensures that only this instance
    of a ParallelRun will be able to register the (param, output) pair it evaluated.
    Uses a lock to handle the optimizer (shared between all ParallelRuns),
    maintains invariant that one ParallelRun will not "suggest" a new point before
    having registered a new evaluated datapoint.
    """

    def __init__(self):
        self.last_param = None
        self.last_output = None

    def step(self, optimizer, f):
        lock.acquire()
        try:
            if self.last_output is not None:
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
    to chosen values (ex: {'batch-size': 32, 'dropout': .1})
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

    return get_metric_value(r)

def parse_params():
    """
    Parse hyperparameter input domains
    """
    tup = args.params
    param_to_min_max = {k: (v[0], v[1]) for k, v in tup.items()}
    return param_to_min_max

def get_metric_value(r):
    metrics = [m[2] for m in r.metrics(metric_name=args.metric, follow=True)]

    if not metrics:
        raise Exception('Run ended prematurely or metric ({0}) was not found (check run logs)'.format(args.metric))

    if args.type == 'average':
        metric_value = sum(metrics) / float(len(metrics))
    elif args.type == 'last':
        metric_value = metrics[-1]
    elif args.type == 'max':
        metric_value = max(metrics)
    else:
        raise Exception('Invalid metric type ({0})'.format(args.type))

    r.wait_status(client.runs.COMPLETE)

    print("\t[{0}] run output: {1} ({2})".format(r.id, round(metric_value, 2), args.metric))

    return metric_value if args.optimize == 'maximize' else (-1)*metric_value

def main():
    #set up optimizer
    pbounds = parse_params()
    # print('pbounds are -> {0}'.format(pbounds))
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1
    )

    #set up args.parallel ParallelRuns
    parallel_runs = []
    for i in range(args.parallel):
        parallel_runs.append(ParallelRun())

    for i in range(int(args.maxruns/args.parallel)):
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


if __name__ == '__main__':
    main()
