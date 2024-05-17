import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from task.TaskLoader import Opt
from ray import tune
import torch

class nn_base(Opt):
    def __init__(self):
        super().__init__()
        
        self.arch = 'torch_nn'
        self.trainer_module = ('models/_baseTrainer.py', 'Trainer')
        
        self.hyper = hyper()
        self.tuner = tuner()
        self.tuning = tuning()
        
        self.base_modify()
        self.hyper_modify()
        self.tuning_modify()
        
        self.task_modify()
        self.ablation_modify()
        
        self.common_process()

    def common_process(self,):
        if "import_path" in self.dict:
            self.import_path = self.import_path.replace(
            '.py', '').replace('/', '.')
        if "trainer_module" in self.dict:
            self.trainer_module = (self.trainer_module[0].replace(
                    '.py', '').replace('/', '.'), self.trainer_module[1])
            
        # if 'gpu' in self.tuner.resource:
        #     num_gpus = torch.cuda.device_count()
        #     trial_perGPU = int(1 // self.tuner.resource['gpu'])
        #     self.tuner.resource['cpu'] = self.tuner.num_cpus
        #     self.tuner.num_cpus = self.tuner.num_cpus * trial_perGPU * num_gpus
            
    
    def base_modify(self,):
        pass
    def hyper_modify(self,):
        pass
    def tuning_modify(self):
        pass
    def ablation_modify(self):
        pass
    def task_modify(self):
        pass



class hyper(Opt):
    def __init__(self):
        super().__init__()
        self.epochs = 100
        self.patience = 10
        self.milestone_step = 3
        self.gamma = 0.5
        self.lr = 0.001

class tuner(Opt):
    def __init__(self):
        super().__init__()
        self.resource = {
            "gpu": 1  # set this for GPUs
        } # Parallel nums = min(num_cpus // cpu, num_gpus // gpu), where num_gpus = torch.cuda.device_count(), which means this setting only affects to the worker scheduler of the ray tuner, and the cpus settings does not affect the system resources the runing worker utilizes.
        
        self.num_samples = 20 # tuner num trails
        self.training_iteration = 100 # max fitness epochs per trail
        self.algo = 'tpe'
        self.num_cpus = os.cpu_count()

class tuning(Opt):
    def __init__(self):
        super().__init__()
        self.lr = tune.loguniform(1e-4, 1e-2)
        self.gamma = tune.uniform(0.33,0.99)
        self.milestone_step = tune.qrandint(1,21,2)
        
        

class AMC_Net_base(nn_base):
    def base_modify(self):
        self.import_path = 'models/AMC_Net.py'
        self.class_name = 'AMC_Net'
    def hyper_modify(self):        
        self.hyper.gamma = 0.1

        self.hyper.extend_channel = 36
        self.hyper.latent_dim = 512
        self.hyper.num_heads = 2
        self.hyper.conv_chan_list = [36, 64, 128, 256]
        
class AWN_base(nn_base):
    def base_modify(self):
        self.import_path = 'models/AWN.py'
        self.class_name = 'AWN'
        self.trainer_module = (self.import_path, 'AWN_Trainer')
        
    def hyper_modify(self):        
        self.hyper.batch_size = 128
        self.hyper.gamma = 0.5
        
        self.hyper.num_level = 1
        self.hyper.regu_details = 0.01
        self.hyper.regu_approx = 0.01
        self.hyper.kernel_size = 3
        self.hyper.in_channels = 64
        self.hyper.latent_dim = 320


class mcldnn_base(nn_base):
    '''Refer to the paper:  'A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition.
    '''
    def base_modify(self):
        self.import_path = 'models/MCLDNN.py'
        self.class_name = 'MCLDNN'
        self.trainer_module = (self.import_path, 'MCLDNN_Trainer')
        
    def hyper_modify(self):
        self.hyper.batch_size = 400
        self.hyper.patience = 60
        self.hyper.milestone_step = 5
        self.hyper.gamma = 0.8
        

        
class vtcnn2_base(nn_base):
    def base_modify(self):
        self.import_path = 'models/VT_CNN2.py'
        self.class_name = 'VTCNN'
        
class cldnn_base(nn_base):
    def base_modify(self):
        self.import_path = 'models/CLDNN.py'
        self.class_name = 'CLDNN'
        
class dualnet_base(nn_base):
    def base_modify(self):
        self.import_path = 'models/Dual_Net.py'
        self.class_name = 'DualNet'

class MCDformer_base(nn_base):
    def base_modify(self):
        self.import_path = 'models/MCDformer.py'
        self.class_name = 'MCDformer'
    def hyper_modify(self):        
        self.hyper.gamma = 0.1

        self.hyper.extend_channel = 36
        self.hyper.latent_dim = 256
        self.hyper.num_heads = 2
        self.hyper.conv_chan_list = [1, 36, 64, 128, 256]


class CTDNN(nn_base):
    def base_modify(self):
        self.import_path = 'models/CTDNN.py'
        self.class_name = 'CTDNN'
    def hyper_modify(self):        
        self.hyper.gamma = 0.1

        self.hyper.num_classes = 11
        self.hyper.encoder_emb = 256
        self.hyper.num_heads = 2
        self.hyper.mlp_ratio = 2
        self.hyper.depth = 2
        self.hyper.patch_size = (2,2)