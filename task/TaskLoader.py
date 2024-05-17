from collections.abc import Mapping
import copy
import numpy as np
import torch
import torch.utils.data as Data
import os
from task.util import set_fitset

class Opt(object):
    def __init__(self, init=None):
        super().__init__()

        if init is not None:
            self.merge(init)

    def merge(self, opts, ele_s=None):
        '''
        Only merge the key-value not in the current Opt.\n
        Using ele_s to select the element in opts to be merged.
        '''
        if isinstance(opts, Mapping):
            new = opts
        else:
            assert isinstance(opts, object)
            new = vars(opts)

        if ele_s is None:
            for key in new:
                if not key in self.dict:
                    self.dict[key] = copy.copy(new[key])
        else:
            for key in ele_s:
                if not key in self.dict:
                    self.dict[key] = copy.copy(new[key])

    def update(self, opts, ignore_unk=False):
        if isinstance(opts, Mapping):
            new = opts
        else:
            assert isinstance(opts, object)
            new = vars(opts)
        for key in new:
            if not key in self.dict and ignore_unk is False:
                raise ValueError(
                    "Unknown config key '{}'".format(key))
            self.dict[key] = copy.copy(new[key])

    @property
    def dict(self):
        '''Gives dict-like access to Params instance by params.dict['learning_rate']'''
        return self.__dict__


class TaskDataset(Opt):
    """
    docstring
    """

    def __init__(self, args=None):
        super().__init__()
        
        
        self.Signals = []
        self.Labels = []
        self.SNRs= []
        self.snrs = []
        self.mods = []
        
        self.num_workers = 0 # used in func: self.load_dataset 
        self.post_data_file = ''
        
        self.rawdata_config()
        self.num_classes = len(self.classes.keys())
        # self.num_snrs = list(np.unique(self.snrs))
        
        # self.load_rawdata()
        # self.pack_dataset()
        self.pack_info()

    def pack_info(self):
        self.info = Opt()
        self.info.merge(self, ['data_name', 'batch_size', 'sig_len', 'num_classes', 'post_data_file'])

    def rawdata_config(self,): 
        """
        Preprocess the raw data, put the Signals, Labels, SNRs, snrs, and mods to the self.rawData, and update the dataset related parameters.
        
        Parameters
        ----------
        data_name: str
        batch_size: int
        sig_len: int
        val_size: float
        test_size: float
        num_classes: int
        classes: dict
        
        """
        self.data_name = 'None'
        self.val_size = 0.2
        self.test_size = 0.2
        self.batch_size = 64
        self.sig_len = 128
        self.num_classes = 0
        self.classes = {}


    def load_rawdata(self, logger = None):
        """
        Loading the rawdata, personalized by args.exp_file in the args.exp_config
        """
        pass
    
    def pack_dataset(self, logger = None):
        """
        Split the preprocessed data, and pack them to self.train_set, self.val_set, self.test_set, self.test_idx
        """
        if os.path.exists(self.info.post_data_file):
            try:
                split_data = torch.load(self.info.post_data_file)
                self.train_set =split_data['train_set']
                self.test_set = split_data['test_set']
                self.val_set = split_data['val_set']
                self.test_idx = split_data['test_idx']
                self.SNRs = split_data['SNRs']
                self.snrs = split_data['snrs']
                self.mods = split_data['mods']
                
                if logger is not None:
                    logger.info(f'Loading pre-split file in the location: {self.info.post_data_file}')
                
            except:
                raise ValueError(f'Error when loading pre-processed datafile in the location: {self.info.post_data_file}')
        
        else:
            Signals, Labels, SNRs, snrs, mods = self.load_rawdata(logger)
            self.SNRs= SNRs
            self.snrs = snrs
            self.mods = mods
            
            if logger is not None:
                logger.info(f'Using the random seed: {self.info.seed}')
                logger.info('Split the dataset to training set, validation set, and test set with the ration of {:.2f}, {:.2f}, and {:.2f}'.format(1- self.test_size - self.val_size, self.val_size, self.test_size))
            
            self.train_set, self.val_set, self.test_set, self.test_idx = self.dataset_Split(Signals=Signals, Labels=Labels, snrs=self.snrs, mods=self.mods, val_size=self.val_size,test_size=self.test_size)
            
            torch.save({
            'train_set': self.train_set,
            'test_set': self.test_set,
            'val_set': self.val_set,
            'test_idx': self.test_idx,
            'SNRs': self.SNRs,
            'snrs': self.snrs,
            'mods':  self.mods  
            }, self.info.post_data_file
            )
        return self.train_set, self.val_set, self.test_set, self.test_idx

        
    @staticmethod
    def dataset_Split(Signals, Labels, snrs, mods, val_size=0.2, test_size=0.2):
        global test_idx
        n_examples = Signals.shape[0]
        n_train = int(n_examples * (1 - val_size - test_size))

        train_idx = []
        test_idx = []
        val_idx = []

        Slices_list = np.linspace(0, n_examples, num=len(mods) * len(snrs) + 1)

        for k in range(0, Slices_list.shape[0] - 1):
            train_idx_subset = np.random.choice(
                range(int(Slices_list[k]), int(Slices_list[k + 1])), size=int(n_train / (len(mods) * len(snrs))),
                replace=False)
            Test_Val_idx_subset = list(
                set(range(int(Slices_list[k]), int(Slices_list[k + 1]))) - set(train_idx_subset))
            test_idx_subset = np.random.choice(Test_Val_idx_subset,
                                               size=int(
                                                   (n_examples - n_train) * test_size / (
                                                       (len(mods) * len(snrs)) * (test_size + val_size))),
                                               replace=False)
            val_idx_subset = list(
                set(Test_Val_idx_subset) - set(test_idx_subset))

            train_idx = np.hstack([train_idx, train_idx_subset])
            val_idx = np.hstack([val_idx, val_idx_subset])
            test_idx = np.hstack([test_idx, test_idx_subset])

        train_idx = train_idx.astype('int64')
        val_idx = val_idx.astype('int64')
        test_idx = test_idx.astype('int64')

        Signals_train = Signals[train_idx]
        Labels_train = Labels[train_idx]

        Signals_test = Signals[test_idx]
        Labels_test = Labels[test_idx]

        Signals_val = Signals[val_idx]
        Labels_val = Labels[val_idx]

        # logger.info(f"Signal_train.shape: {list(Signals_train.shape)}", )
        # logger.info(f"Signal_val.shape: {list(Signals_val.shape)}")
        # logger.info(f"Signal_test.shape: {list(Signals_test.shape)}")
        # logger.info('*' * 20)

        return (Signals_train, Labels_train), \
            (Signals_val, Labels_val), \
            (Signals_test, Labels_test), \
            test_idx


    def load_fitset(self, fit_batch_size = None):
        _fit_batch_size = fit_batch_size if fit_batch_size is not None else self.batch_size
        train_loader, val_loader = set_fitset(batch_size= _fit_batch_size, num_workers=self.num_workers, train_set=self.train_set, val_set=self.val_set)

        return train_loader, val_loader
    
    def load_testset(self, test_batch_size = 64):
        Signals_test, Labels_test = self.test_set
        
        Sample_list = []
        Label_list = []
        
        if 'num_snrs' not in self.dict:
            self.num_snrs = list(np.unique(self.snrs))
        
        
        for snr in self.num_snrs:
            test_SNRs = map(lambda x: self.SNRs[x], self.test_idx)
            test_SNRs = list(test_SNRs)
            test_SNRs = np.array(test_SNRs).squeeze()
            test_sig_i = Signals_test[np.where(np.array(test_SNRs) == snr)]
            test_lab_i = Labels_test[np.where(np.array(test_SNRs) == snr)]
            Sample = torch.chunk(test_sig_i, test_batch_size, dim=0)
            Label = torch.chunk(test_lab_i, test_batch_size, dim=0)
            
            Sample_list.append(Sample)
            Label_list.append(Label)
        
        return Sample_list, Label_list