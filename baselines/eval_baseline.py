"""
Unified baseline evaluation script

Simon Bing
ETHZ 2020
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from disentanglement_lib.evaluation import evaluate
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('base_dir', '', 'Base directory')
flags.DEFINE_string('exp_name', '', 'Name of experiment to evaluate')
flags.DEFINE_enum('model', 'adagvae', ['adagvae', 'annealedvae', 'betavae',
                                       'betatcvae', 'dipvae_i', 'dipvae_ii',
                                       'factorvae'], 'Model to evaluate')
flags.DEFINE_enum('data', 'dsprites', ['dsprites', 'smallnorb', 'cars3d', 'shapes3d'],
                  'Dataset for evaluation')
flags.DEFINE_string('subset', None, 'Subset of dataset')
flags.DEFINE_enum('metric', 'dci', ['dci', 'mig', 'modularity', 'sap'], 'Evaluation metric')

flags.mark_flag_as_required('subset')

def main(argv):
    baseline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 FLAGS.model)
    result_path = os.path.join(baseline_path, FLAGS.base_dir, FLAGS.exp_name, 'metrics', FLAGS.metric)
    representation_path = os.path.join(baseline_path, FLAGS.base_dir, FLAGS.exp_name, 'representation')

    gin_bindings = [
        "dataset.name = '{}'".format(FLAGS.data),
        "subset.name = '{}'".format(FLAGS.subset)
    ]

    evaluate.evaluate_with_gin(representation_path, result_path, True,
                               [os.path.join(baseline_path, F'{FLAGS.model}_{FLAGS.metric}.gin')], gin_bindings)

if __name__ == '__main__':
    app.run(main)