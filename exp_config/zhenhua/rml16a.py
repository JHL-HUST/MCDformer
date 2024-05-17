import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["RAY_memory_usage_threshold"] = "0.99"
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from task.TaskParser import get_parser
from task.TaskWrapper import Task
from task.TaskLoader import TaskDataset

import pickle
import numpy as np
import torch

from ray import tune

from models._baseSetting import AMC_Net_base, AWN_base, mcldnn_base, vtcnn2_base, dualnet_base,MCDformer_base,cldnn_base,CTDNN

class Data(TaskDataset):
    def __init__(self, opts):
        '''Merge the input args to the self object'''
        super().__init__(opts)
     
    def rawdata_config(self) -> object:
        self.data_name = 'RML2016.10a'
        self.batch_size = 64
        self.sig_len = 128
        
        self.val_size = 0.2
        self.test_size = 0.2
        
        self.classes = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}
        
        self.post_data_file = 'Dataset/RML2016.10a/RML2016.10a_dict.split.pt'
        
    def load_rawdata(self, logger = None):
        file_pointer = 'Dataset/RML2016.10a/RML2016.10a_dict.pkl'
        
        if logger is not None:
            logger.info('*'*80 + '\n' +f'Loading raw file in the location: {file_pointer}')
        
        Signals = []
        Labels = []
        SNRs = []
        
        Set = pickle.load(open(file_pointer, 'rb'), encoding='bytes')
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Set.keys())))), [1, 0])
        for mod in mods:
            for snr in snrs:
                Signals.append(Set[(mod, snr)])
                for i in range(Set[(mod, snr)].shape[0]):
                    Labels.append(mod)
                    SNRs.append(snr)
                    
        Signals = np.vstack(Signals)
        Signals = torch.from_numpy(Signals.astype(np.float32))

        Labels = [self.classes[i] for i in Labels]  # mapping modulation formats(str) to int
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)
        
        return Signals, Labels, SNRs, snrs, mods

class amcnet(AMC_Net_base):
    def task_modify(self):
        self.hyper.extend_channel = 36
        self.hyper.latent_dim = 512
        self.hyper.num_heads = 2
        self.hyper.conv_chan_list = [36, 64, 128, 256]        
        self.hyper.pretraining_file = 'pretraining_models/pretrain_models_16a/RML2016.10a_AMC_Net.best.pt'
        # 'data/RML2016.10a/pretrain_models/2016.10a_AMC_Net.best.pt'
    
class awn(AWN_base):
    def task_modify(self):
        self.hyper.num_level = 1
        self.hyper.regu_details = 0.01
        self.hyper.regu_approx = 0.01
        self.hyper.kernel_size = 3
        self.hyper.in_channels = 64
        self.hyper.latent_dim = 320    
        self.hyper.pretraining_file = 'pretraining_models/pretrain_models_16a/RML2016.10a_AWN.best.pt'
        
        # self.tuner.resource = {
        #     "cpu": 5,
        #     "gpu": 0.5  # set this for GPUs
        # }
    
class vtcnn(vtcnn2_base):
    def task_modify(self):
        self.hyper.epochs = 100
        self.hyper.patience = 10
        self.hyper.gamma = 0.5
        self.hyper.pretraining_file = 'pretraining_models/pretrain_models_16a/RML2016.10a_CLDNN.best.pt'
        # self.tuner.resource = {
        #     "cpu": 5,
        #     "gpu": 0.5  # set this for GPUs
        # }

class dualnet(dualnet_base):
    def task_modify(self):
        self.hyper.epochs = 200
        self.hyper.patience = 15
        self.hyper.pretraining_file = 'pretraining_models/pretrain_models_16a/RML2016.10a_DualNet.best.pt'

class mcl(mcldnn_base):
    def task_modify(self):
        self.hyper.epochs = 100
        self.hyper.pretraining_file = 'pretraining_models/pretrain_models_16a/RML2016.10a_MCLDNN.best.pt'

        # self.tuner.num_samples = 40
        # self.tuner.training_iteration = self.hyper.epochs
        # # self.tuner.num_cpus = 32 * 3
        # self.tuner.resource = {
        #     "gpu": 1  # set this for GPUs
        # }
        
        # self.tuner.points_to_evaluate=[{
        #     'lr':0.001,
        #     'gamma':0.8,
        #     'milestone_step':5,
        #     'batch_size': 400
        # }]
        
        # # self.tuner.using_sched = False
        # self.tuning.lr = tune.loguniform(1e-4, 1e-2)
        # self.tuning.gamma = tune.uniform(0.5,0.99)
        # self.tuning.milestone_step = tune.qrandint(2,10,1)
        # self.tuning.batch_size = tune.choice([64, 128, 192, 256, 320, 400])

class MCDformer(MCDformer_base):
    def ablation_modify(self):
        self.hyper.epochs = 200
        self.hyper.pretraining_file = ''
        self.hyper.num_heads = 2
        self.hyper.lr = 0.00054
        self.hyper.gamma = 0.3254
        self.hyper.milestone_step = 13
        self.hyper.batch_size = 32

        # self.tuning.lr = tune.loguniform(1e-4, 1e-2)
        # self.tuning.gamma = tune.uniform(0.1,0.9)
        # self.tuning.milestone_step = tune.qrandint(1,20,2)
        # self.tuner.algo = 'bayes'
        # # self.tuning.num_heads = tune.choice([1,2,4])
        # self.tuning.batch_size = tune.choice([32, 64, 128])

        # self.tuner.resource = {
        #     # "cpu": 3,
        #     "gpu": 1,
        # } 
        # self.tuner.num_samples = 1

class ctdnn(CTDNN):
    def ablation_modify(self):
        self.hyper.epochs = 100
        self.hyper.pretraining_file = 'exp_tempTest/RML2016.10a/train2.ctdnn/fit/ctdnn/checkpoint/RML2016.10a_ctdnn.best.pt'
        self.hyper.num_heads = 2
        self.hyper.gamma = 0.5
        self.hyper.batch_size = 400
        self.hyper.lr = 1e-3
        self.hyper.patch_size = (2,8)
        # self.tuning.lr = tune.loguniform(1e-4, 1e-2)
        # self.tuning.gamma = tune.uniform(0.5,0.99)
        # self.tuning.milestone_step = tune.qrandint(1,20,2)
        # # self.tuner.algo = 'tpe'
        # self.tuning.num_heads = tune.choice([1,2,4])
        # self.tuning.batch_size = tune.choice([32, 64, 128, 192, 256])

        # self.tuner.resource = {
        #     # "cpu": 3,
        #     "gpu": 0.5,
        # } 
        # self.tuner.num_samples = 40   
class cldnn(cldnn_base):
    def ablation_modify(self):
        self.hyper.pretraining_file = 'pretraining_models/pretrain_models_16a/RML2016.10a_CLDNN.best.pt'
if __name__ == "__main__":
    args = get_parser()
    args.cuda = True
    
    args.exp_config = os.path.dirname(sys.argv[0]).replace(os.getcwd()+'/', '')
    args.exp_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    # args.exp_name = 'icassp23'
    # amcnet awn vtcnn dualnet mcl ctdnn MCDformer
    model_name = 'MCDformer'
    args.exp_name = f'test_train_{model_name}' # tuning.infv3
    args.gid = 0
    
    args.test = True
    args.clean = True
    args.model = model_name # dualnet
    
    task = Task(args)

    # train
    task.conduct()

    # evaluate
    task.evaluate(force_update=True)

    # parameter tuning 
    # task.tuning()

            
    