import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import torch
import torch.utils.data as Data

import ray
from task.TaskLoader import Opt
# from ray.tune.suggest.bohb import TuneBOHB
# from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import PopulationBasedTraining 
# https://arxiv.org/pdf/1711.09846.pdf.
from ray.tune.schedulers.pb2 import PB2 
# pip install GPy sklearn
# https://arxiv.org/abs/2002.02518 (NIPS 2020)
from ray import tune
from ray.air import session, FailureConfig
from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig
# from ray.air.checkpoint import Checkpoint

from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search.ax import AxSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune import ExperimentAnalysis

import nevergrad as ng

import importlib

from task.util import os_rmdirs,set_logger,set_fitset


class HyperTuner(Opt):
    def __init__(self, opts = None, logger= None, subPack = None):
        super().__init__()

        if opts is not None:
            self.merge(opts)

        self.points_to_evaluate = []
        if 'points_to_evaluate' in self.tuner.dict:
            self.points_to_evaluate = self.tuner.points_to_evaluate
            assert len(self.points_to_evaluate) > 0

        self.best_config = Opt()
        
        self.logger = logger
        
        # train_loader, val_loader = subPack.load_fitset()
        # self.batch_size = subPack.batch_size
        self.subPack = subPack
        # self.train_data = subPack.train_set
        # self.valid_data = subPack.val_set
        
        self.metric = 'best_val_acc'
        if 'num_samples' not in self.tuner.dict:
            self.tuner.num_samples = 20
            if self.algo_name == 'grid':
                self.tuner.num_samples = 1
                
        if 'num_cpus' not in self.tuner.dict:
            self.tuner.num_cpus = 30
        
        if 'training_iteration' not in self.tuner.dict:
            self.tuner.training_iteration = 100
        
        self.using_sched = True
        if 'using_sched' in self.tuner.dict and self.tuner.using_sched is False:
            self.using_sched = False
        
        if 'resource' not in self.tuner.dict:
            self.resource = {
            "gpu": 1  # set this for GPUs
        }
        else:
            self.resource = self.tuner.resource
            
        # Parallel nums = min(num_cpus // cpu, num_gpus // gpu), where num_gpus = torch.cuda.device_count()
            
        self.loss_lower_bound = 0
        self.algo_init()
    
                  
    def algo_init(self,):
        if 'algo' not in self.tuner.dict:
            self.algo_name = 'rand'
        elif self.tuner.algo not in ['bayes','tpe','pso', 'rand', 'grid']:
            raise ValueError('Non supported tuning algo: {}'.format(self.tuner.algo))
        else:
            self.algo_name = self.tuner.algo
        
        if self.algo_name == 'bayes':
            self.tuner.name = 'Bayes_Search'
            self.algo_func = ConcurrencyLimiter(AxSearch(
                # metric=self.metric, mode='max',
                verbose_logging = False), max_concurrent=10)
            
        elif self.algo_name == 'tpe':
            # Tree-structured Parzen Estimator https://docs.ray.io/en/master/tune/examples/optuna_example.html
            self.tuner.name = 'TPE_Search'
            self.algo_func =  ConcurrencyLimiter(
            OptunaSearch(points_to_evaluate=self.points_to_evaluate
                ), 
            max_concurrent=4
            )
            
        elif self.algo_name == 'pso':
            # https://github.com/facebookresearch/nevergrad
            self.tuner.name = 'PSO_Search'
            _popsize= min((20, self.tuner.num_samples // 10))
            self.algo_func = NevergradSearch(
            optimizer=ng.optimizers.ConfiguredPSO(
                popsize= _popsize
                ),
            # metric=self.metric,
            # mode="max",
            points_to_evaluate=self.points_to_evaluate
            )
        elif self.algo_name == 'rand' or self.algo_name == 'grid':
            self.tuner.name = 'Rand_Search'   
            self.algo_func = BasicVariantGenerator(max_concurrent=4)

    def once_sample(self,):
        config = Opt(init=self.tuning)
        for key in self.tuning.dict:
            config.dict[key] = self.tuning.dict[key].sample()
        
        _hyper = Opt()
        _hyper.merge(self.hyper)
        _hyper.update(config) # Using ignore_unk will be very risky
        model = importlib.import_module(self.import_path)
        model = getattr(model, self.class_name)
        model = model(_hyper, self.logger)
        
        train_loader, val_loader = self.subPack.load_fitset(_hyper.batch_size)
        fit_info = model.xfit(train_loader, val_loader)
        t_acc, v_acc = fit_info.train_acc.max(), fit_info.val_acc.max()   
        
        metric_dict = {
            'tra_acc': t_acc,
            'val_acc': v_acc,
        }
        self.best_result = metric_dict[self.metric]
        self.best_config.merge(config)
        return self.best_config

    def conduct(self,):
        self.best_checkpoint_path = None
        
        if self.tuner.num_samples == 1 and self.algo_name == 'rand':
            self.best_config = self.once_sample()
        else:
            tuner_algo_dir = os.path.join(self.tuner.dir, self.algo_name)
            results_pkl = os.path.join(tuner_algo_dir, 'tuner.pkl' )
            if os.path.exists(results_pkl):
                try:
                    analysis = ExperimentAnalysis(tuner_algo_dir)
                    best_config = analysis.get_best_config(metric='val_acc', mode='max', scope='all')
                    # best_checkpoint = analysis.get_best_checkpoint(analysis.get_best_logdir(metric='val_acc', mode='max', scope = 'all'), metric='val_acc', mode='max')
                    best_trail = analysis.get_best_trial(metric='val_acc', mode='max', scope='all')
                    best_checkpoint = analysis.get_best_checkpoint(best_trail, metric='val_acc', mode='max')
                    self.best_checkpoint_path = os.path.join(best_checkpoint.path, 'model.pth')
                    
                    self.best_config  = best_config
                except:
                    raise ValueError('Error in loading the tuning results in {}\nPlease check the tuning results carefully, then remove it and rerun.'.format(tuner_algo_dir))
                    # raise SystemExit()
                    # os_rmdirs(tuner_algo_dir) 
                    # self._conduct(tuner_algo_dir)
            else:
                self._conduct()
            # self.best_config = self._conduct()
        
        return self.best_config, self.best_checkpoint_path 

    def _conduct(self,):
        
        func_data = Opt()
        # func_data.logger = self.logger
        func_data.merge(self,['hyper', 'import_path', 'class_name', 'trainer_module'])
        func_data.merge(self.subPack, ['train_set', 'val_set'])
        
        # ray.init(num_cpus=self.tuner.num_cpus)
        os.environ['RAY_COLOR_PREFIX'] = '1'
        ray.init()
        sched = ASHAScheduler(time_attr='training_iteration', max_t=self.tuner.training_iteration, grace_period= 20) if self.using_sched else None
        # self.tuner.num_samples = 80
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(TuningCell, data=func_data), 
                resources=self.resource
            ),
            param_space=self.tuning.dict,
            tune_config=tune.TuneConfig(
                # name=self.algo_name,
                search_alg=self.algo_func,
                # resources_per_trial=self.resource,
                metric='best_val_acc',
                mode="max",
                num_samples=self.tuner.num_samples,
                scheduler=sched,
                # verbose=1,
                # raise_on_failed_trial = False
            ),
            run_config=RunConfig(
                name=self.algo_name,
                storage_path=self.tuner.dir,
                verbose=1,
                failure_config=FailureConfig(max_failures=self.tuner.num_samples // 2),
                stop={'training_iteration':self.tuner.training_iteration,
                      'stop_counter': self.hyper.patience},
                checkpoint_config=CheckpointConfig(
                    num_to_keep=3,
                    checkpoint_score_attribute ='val_acc',
                    checkpoint_score_order='max',
                    checkpoint_frequency= 1
                )      
            )
        )
        
        results = tuner.fit() 
        # https://docs.ray.io/en/latest/tune/tutorials/tune-output.html?highlight=tensorboard#where-to-find-log-to-file-files
        # to see this,  using:  tensorboard --logdir /exp_tempTest/RML2016.10a/tuning.mcl/fit/mcl/tuner/tpe --host 192.168.80.XXX
        
        df = results.get_dataframe()
        df.to_csv(os.path.join(self.tuner.dir, '{}.trial.csv'.format(self.algo_name)))
        ray.shutdown()
        
        best_result = results.get_best_result(self.metric, 'max', scope='all')
        self.best_config.merge(best_result.config)
        self.best_result = best_result.metrics
        self.best_checkpoint_path = os.path.join(best_result.checkpoint.path, 'model.pth')
      
    

class TuningCell(tune.Trainable):
    '''
    Trainable class of tuning stochastic neural network
    see https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html#the-train-function
    '''
    def setup(self, config, data=None):
        self.train_data = data.train_set
        self.valid_data = data.val_set
        self.base_hyper = data.hyper
        self.import_path = data.import_path
        self.class_name = data.class_name
        # self.logger = data.logger
        
        logger_path = os.path.join(self.logdir, '{}.log'.format(self.class_name))
        self.logger = set_logger(logger_path, f'{self.class_name}.tuning', rewrite=False)
        
        
        _hyper = Opt()
        _hyper.merge(self.base_hyper)
        _hyper.update(config) # Using ignore_unk will be very risky
        
        
        self.sample_hyper = _hyper
        self.num_workers = 0 if 'num_workers' not in _hyper.dict else  _hyper.num_workers
        train_loader, val_loader = set_fitset(batch_size=_hyper.batch_size, train_set=data.train_set, val_set=data.val_set)
        
        self.logger.critical('Loading training set and validation set.')
        self.logger.info(f"Train_loader batch: {len(train_loader)}")
        self.logger.info(f"Val_loader batch: {len(val_loader)}")
        self.logger.critical('>'*40)
        
        _model = importlib.import_module(self.import_path)
        _model = getattr(_model, self.class_name)
        sample_model = _model(self.sample_hyper, self.logger)
        
        self.logger.critical('Loading Model.')
        self.logger.critical(f'Model: \n{str(sample_model)}')
        self.logger.info(">>> Total params: {:.2f}M".format(
                    sum(p.numel() for p in list(sample_model.parameters())) / 1000000.0))
        self.logger.critical('Start fit.')
            
        trainer = importlib.import_module(data.trainer_module[0])
        trainer = getattr(trainer, data.trainer_module[1])
        
        self.trainer = trainer(sample_model, train_loader, val_loader, self.base_hyper, self.logger)
        # self.trainer.cfg.patience = self.trainer.cfg.epochs # Actually, in TuningCell, patience does not work as the same as in trainer.
        self.trainer.before_train()
    
    def load_fitset(self, train_set = None, val_set =None, fit_batch_size = None):

        _fit_batch_size = fit_batch_size if fit_batch_size is not None else self.batch_size

        train_data = Data.TensorDataset(*train_set)
        val_data = Data.TensorDataset(*val_set)

        train_loader = Data.DataLoader(
            dataset=train_data,
            batch_size=_fit_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        val_loader = Data.DataLoader(
            dataset=val_data,
            batch_size=_fit_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        # logger.info(f"train_loader batch: {len(train_loader)}")
        # logger.info(f"val_loader batch: {len(val_loader)}")

        return train_loader, val_loader
    
    def step(self,):
        self.trainer.before_train_step()
        _, t_acc = self.trainer.run_train_step(ray=True)
        self.trainer.after_train_step()
        self.trainer.before_val_step()
        _, v_acc = self.trainer.run_val_step(ray=True)
        self.trainer.after_val_step(checkpoint = False)
        self.trainer.iter += 1      
        
        best_acc =  self.trainer.best_monitor
        
        return {
            'tra_acc': t_acc,
            'val_acc': v_acc,
            'best_val_acc': best_acc * 100, 
            'stop_counter': self.trainer.early_stopping.counter,
        }
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.trainer.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.trainer.model.load_state_dict(torch.load(checkpoint_path)) 