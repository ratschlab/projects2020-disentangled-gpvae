"""
Get training time of experiments

Simon Bing,
ETHZ 2020
"""

import numpy as np
import os
from glob import glob
import json
from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'gpvae', 'Model for which dci score should be calculated')
flags.DEFINE_string('base_dir', '', 'base directory of models')
flags.DEFINE_list('params', [64], 'Parameters tested in experiment')
flags.DEFINE_string('exp_name', '', 'Experiment naming scheme')
flags.DEFINE_boolean('save', False, 'Save aggregated scores')

def walklevel(some_dir, level=0):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def aggregate_gpvae(N, params, base_dir='dsprites_dim_'):
    """
    Collects all dci scores and aggregates into single array.
    Args:
        N, int:             Number of random seeds
        params, list:       params tested
        base_dir, string:   Base directory naming scheme

    Returns:
        dci_scores, [3xNxM] np array
    """
    dci_scores = np.zeros((3,N,len(params)), dtype=np.float32)

    for m, param in enumerate(params):
        param_dir = os.path.join(base_dir, '{}_{}'.format(FLAGS.exp_name, param))

        models_path = os.path.join('models', param_dir)
        print(models_path)
        for _, dirs, _ in os.walk(models_path):
            for n, dir in enumerate(dirs):
                for _, _, files in os.walk(os.path.join(models_path, dir)):
                    for filename in files:
                        if filename.startswith('dci'):
                            # print(n, filename)
                            dci = np.load(os.path.join(models_path, dir, filename)) # Might need abspath here
                            dci_scores[0,n,m] = dci['disentanglement']
                            dci_scores[1,n,m] = dci['completeness']
                            dci_scores[2,n,m] = dci['informativeness_test']

    return dci_scores

def get_time_baseline(N, params, base_dir='dim'):
    """
    Collects all dci scores and aggregates into single array.
    Args:
        N, int:             Number of random seeds
        params, list:       params tested
        base_dir, string:   Base directory naming scheme

    Returns:
        train_times, [NxM] np array
    """
    train_times = np.zeros((N,len(params)), dtype=np.float32)

    for m, param in enumerate(params):
        param_dir = os.path.join(base_dir, '{}_{}'.format(FLAGS.exp_name, param))
        model_path = os.path.join('baselines', FLAGS.model, param_dir)

        for _, dirs, _ in walklevel(model_path):
            for n, dir in enumerate(dirs):
                # print(n, dir)
                json_path = os.path.join(model_path, dir, 'metrics', 'dci',
                                         'results', 'aggregate',
                                         'train.json')
                with open(json_path) as json_file:
                    train_meta = json.load(json_file)  # PROPERLY PARSE JSON FILE
                    train_times[n, m] = train_meta['train_results.elapsed_time']

        print(F'Avg time {param}: {np.mean(train_times[:,m])} ')

    print(F'Train times shape: {train_times.shape}')


    return train_times

def main(argv):
    del argv # Unused

    n_experiments = 10

    if FLAGS.model == 'gpvae':
        dci_scores = aggregate_gpvae(n_experiments, FLAGS.params, FLAGS.base_dir)
    elif FLAGS.model in ['annealedvae', 'betavae', 'betatcvae', 'factorvae', 'dipvae_i', 'dipvae_ii']:
        dci_scores = get_time_baseline(n_experiments, FLAGS.params, FLAGS.base_dir)
    else:
        raise ValueError("Model must be one of: ['gpvae', 'annealedvae', 'betavae', 'betatcvae', 'factorvae', 'dipvae_i', 'dipvae_ii']")

    print(dci_scores.shape)

    if FLAGS.save:
        if FLAGS.model == 'gpvae':
            np.save(os.path.join('models', FLAGS.base_dir, 'dci_aggr.npy'), dci_scores)
            print(F"Saved scores at :{os.path.join('models', FLAGS.base_dir, 'dci_aggr.npy')}")
        elif FLAGS.model in ['annealedvae', 'betavae', 'betatcvae', 'factorvae', 'dipvae_i', 'dipvae_ii']:
            np.save(os.path.join('baselines', FLAGS.model, FLAGS.base_dir, 'dci_aggr.npy'), dci_scores)
            print(F"Saved scores at :{os.path.join('baselines', FLAGS.model, FLAGS.base_dir, 'dci_aggr.npy')}")
        else:
            raise ValueError(
                "Model must be one of: ['gpvae', 'annealedvae', 'betavae', 'betatcvae', 'factorvae', 'dipvae_i', 'dipvae_ii']")

if __name__ == '__main__':
    app.run(main)