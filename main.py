# python main.py --phase train --dataset sun2rain --batch_size 1

from MUNIT import MUNIT
import argparse
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of MUNIT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
    opts = parser.parse_args()

    config = get_config(opts.config)

    return check_args(config)

"""checking arguments"""
def check_args(config):
    # --checkpoint_dir
    check_folder(config['checkpoint_dir'])

    # --result_dir
    check_folder(config['result_dir'])

    # --result_dir
    check_folder(config['log_dir'])

    # --sample_dir
    check_folder(config['sample_dir'])

    # --epoch
    try:
        assert config['epoch'] >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert config['batch_size'] >= 1
    except:
        print('batch size must be larger than or equal to one')
    return config

"""main"""
def main():
    # parse arguments
    conf = parse_args()
    if conf is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = MUNIT(sess, conf)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if conf['phase'] == 'train' :
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

        if conf['phase'] == 'test' :
            gan.test()
            print(" [*] Test finished!")

        if conf['phase'] == 'guide' :
            gan.style_guide_test()
            print(" [*] Guide finished!")

if __name__ == '__main__':
    main()