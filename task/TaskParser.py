import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import argparse


def get_parser(parsing = True):
    """
    Generate a parameters parser.
    """
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    parser = argparse.ArgumentParser(description="Automatic Modulation Classification with pytorch")

    # -----------------------------------------------------------------------------------
    # Model name
    parser.add_argument('-model', type=str, default='mlp', help='name of the implemented model')
    parser.add_argument('--seed', type=int, default=2022)

    # -----------------------------------------------------------------------------------
    # experimental location parameters
    parser.add_argument('-exp_config', type=str, default='exp_config', help='folder name of the dataset')
    parser.add_argument('-exp_file', type=str, default='demo',help='file name of the dataset')
    parser.add_argument('-exp_name', type=str, default='unitTest', metavar='N',
                        help='exp_name  (default: mimo)')
    parser.add_argument('-tag',type=str, default='', help='additional experimental model tag')

    # -----------------------------------------------------------------------------------
    # experimental log parameters
    parser.add_argument('-test', default=False,action='store_true',
                        help='Whether to use unitTest mode')
    parser.add_argument('-clean', default=True, action='store_true',
                        help='Whether to clean the unitTest folder after each trail')    
    parser.add_argument('-logger_level', type=int, default=20, help='experiment log level')

    # -----------------------------------------------------------------------------------
    # experiment repetitive times
    parser.add_argument('-rep_times', type=int, default=1, help='experiment repetitive times')

    parser.add_argument('-cuda',default=False, action='store_true', help='experiment with cuda')
    parser.add_argument('-gid',type=int, default=0, help='default gpu id')

    # -----------------------------------------------------------------------------------
    # tune resource parameters
    parser.add_argument('-tune', default=False, help='execute tune or not')
    # parser.add_argument('-cores', type=int, default=2, help='cpu cores per trial in tune')
    # parser.add_argument('-cards',type=int, default=0.25, help='gpu cards per trial in tune')
    # parser.add_argument('-tuner_iters', type=int, default=50, help='hyper-parameter search times')
    # parser.add_argument('-tuner_epochPerIter',type=int,default=1)
    
    if parsing:
        args = parser.parse_args()
        return args
    else:
        return parser