import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from tqdm.std import tqdm
import importlib

from task.TaskLoader import Opt
from task.TaskParser import get_parser

import torch
import numpy as np
import statistics

from tqdm import tqdm
import shutil
from task.util import os_makedirs, os_rmdirs, set_logger, fix_seed
from task.TaskVisual import save_training_process, save_confmat, save_snr_acc
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
from collections.abc import Mapping

from task.TaskTuner import HyperTuner

class Task(Opt):
    def __init__(self, args):
        # self.opts = Opt()
        # self.opts.merge(args)
        self.seed = args.seed
        fix_seed(args.seed)

        self.exp_module_path = importlib.import_module('{}.{}'.format(
            args.exp_config.replace('/', '.'), args.exp_file))
        

        self.data_config(args)
        self.model_config(args)
        self.exp_config(args)
        self.data_statue = False
        self.fit_statue = False
        self.tune = False
        
    def data_config(self, args):
        # data_opts = getattr(self.exp_module_path, args.dataset + '_data')
        data_opts = getattr(self.exp_module_path, 'Data')
        self.data_opts = data_opts(args)
        self.data_name = self.data_opts.data_name
        
    def model_config(self, args):
        self.model_name = args.model

        # load the specifical config firstly, if not exists, load the common config
        if hasattr(self.exp_module_path, self.model_name):
            model_opts = getattr(self.exp_module_path,
                                 args.model)
        else:
            try:
                share_module_path = importlib.import_module('models._baseSetting')
                model_opts = getattr(
                    share_module_path, self.model_name + '_base')
            except:
                raise ValueError(
                    'Non-supported model "{}" in the "{}" module, please check the module or the model name'.format(self.model_name, self.exp_module_path))

        self.model_opts = model_opts()
        self.model_opts.hyper.merge(opts=self.data_opts.info) #   Only merge the key-value not in the current hyper.\n
        
        if 'hyper' in vars(args):
            self.model_opts.hyper.update(args.hyper)


    def exp_config(self, args):
        cuda_exist = torch.cuda.is_available()
        if cuda_exist and args.cuda:
            self.model_opts.hyper.device = torch.device('cuda:{}'.format(args.gid))
        else:
            self.model_opts.hyper.device = torch.device('cpu')


        self.exp_dir = 'exp_results' if args.test == False else 'exp_tempTest'

        self.exp_dir = os.path.join(self.exp_dir, self.data_name)
        
        if args.exp_name is not None:
            self.exp_dir = os.path.join(self.exp_dir, args.exp_name)

        self.fit_dir = os.path.join(self.exp_dir, 'fit')
        self.eval_dir = os.path.join(self.exp_dir, 'eval')

        self.model_name = '{}'.format(args.model) if args.tag == '' else '{}_{}'.format(args.model, args.tag)

        self.model_fit_dir = os.path.join(self.fit_dir, self.model_name)
        self.model_pred_dir = os.path.join(self.model_fit_dir, 'pred_results')
        self.model_result_file = os.path.join(self.model_pred_dir, 'results.npz')
        
        if args.test and args.clean:
            os_rmdirs(self.model_fit_dir)
            
        if args.test and args.logger_level != 20:
            self.logger_level = 50  # equal to critical
        else:
            self.logger_level = 20  # equal to info

        # self.rep_times = args.rep_times

        # self.cid_list = args.cid

        # if self.cid_list == ['all']:
        #     self.cid_list = list(range(self.rep_times))
        # else:
        #     _temp = [int(c) for c in self.cid_list]
        #     self.cid_list = _temp


            # self.tune = args.tune  # default False
            
    def model_import(self,):
        model = importlib.import_module(self.model_opts.import_path)
        model = getattr(model, self.model_opts.class_name)
        return model        
    
    def logger_config(self, dir, stage):
        log_path = os.path.join(dir, 'logs',
                                '{}.{}.log'.format(stage,self.data_name))
        log_name = '{}.{}'.format(
            self.data_name, self.model_name)
        logger = set_logger(log_path, log_name, self.logger_level)
        return logger
    
    def load_data(self, logger= None):
        """
        docstring
        """
        if logger is None:
            logger = self.logger_config(self.model_fit_dir, 'data')
        logger.critical(
            'Loading the datasets {}'.format(self.data_name))
        self.data_opts.info.seed = self.seed
            
        # self.data_opts.load_rawdata(logger = logger)
        self.data_opts.pack_dataset(logger = logger)
        
        self.data_opts.num_snrs = list(np.unique(self.data_opts.snrs))
        # logger.critical('-'*80)
        Dataset_size = self.data_opts.train_set[0].shape[0] + self.data_opts.val_set[0].shape[0] + self.data_opts.test_set[0].shape[0]
        logger.critical('-'*80) 
        logger.critical(f'Dataset size: {int(Dataset_size)}\t Class num: {self.data_opts.num_classes}')
        logger.critical(f"Signal_train.shape: {list(self.data_opts.train_set[0].shape)}", )
        logger.critical(f"Signal_val.shape: {list(self.data_opts.val_set[0].shape)}")
        logger.critical(f"Signal_test.shape: {list(self.data_opts.test_set[0].shape)}")
        self.data_statue = True
        
    
    def conduct(self, force_update = False):
        # os_makedirs(self.model_fit_dir)        
        # for i in tqdm(self.cid_list):
        if not os.path.exists(self.model_result_file) or force_update:     
            os_makedirs(self.model_pred_dir)
        
            task_logger= self.logger_config(
                    self.model_fit_dir, 'train')
            self.load_data(logger=task_logger)
            
            self.conduct_fit(clogger= task_logger,result_file = self.model_result_file)
            
            self.fit_statue = True
            
            self.evaluate(elogger= task_logger)
        else:
            self.evaluate()
  
            

    def conduct_fit(self, clogger = None, result_file = None, innerSaving = True):
        try:
            if clogger is None:
                clogger = self.logger_config(
                    self.model_fit_dir, 'train')
                        
            clogger.critical('*'*80)
            clogger.critical('Dataset: {}\t Model:{} \t Class: {}'.format(
                self.data_name, self.model_name, self.data_opts.num_classes))
            
            cid_hyper = Opt(self.model_opts.hyper)
            cid_hyper.num_classes = self.data_opts.num_classes
            cid_hyper.model_fit_dir = self.model_fit_dir
            cid_hyper.model_name = self.model_name
            cid_hyper.data_name = self.data_name     
            # cid_hyper.cid = i
            
            #toDo: set Tune, and loading the best parameters
            if self.tune:
                if 'best_hyper' not in self.dict:
                    self.tuning()
                    
                best_hyper = self.best_hyper
                cid_hyper.update(best_hyper)
                clogger.info("Updating tuning result complete.")
                clogger.critical('-'*80)
                if self.best_checkpoint_path is not None:
                    cid_hyper.pretraining_file = self.best_checkpoint_path

            

            train_loader, val_loader = self.data_opts.load_fitset(fit_batch_size = cid_hyper.batch_size)
            clogger.critical('Loading training set and validation set.')
            clogger.info(f"Train_loader batch: {len(train_loader)}")
            clogger.info(f"Val_loader batch: {len(val_loader)}")
            clogger.critical('>'*40)
            
            
            model = self.model_import()
            model = model(cid_hyper, clogger) #todo: check model_fit_dir in model and model trainer
            clogger.critical('Loading Model.')
            clogger.critical(f'Model: \n{str(model)}')

            if self.model_opts.arch == 'torch_nn':
                clogger.info(">>> Total params: {:.2f}M".format(
                    sum(p.numel() for p in list(model.parameters())) / 1000000.0))         
    
            clogger.critical('Start fit.')
            epochs_stats =  model.xfit(train_loader, val_loader) 
            # epochs_stats is a dataframe with the following columns at least
            # pd.DataFrame(
            # data={"lr_list": self.lr_list,
            #       "train_loss": self.train_loss_list,
            #       "val_loss": self.val_loss_list,
            #       "train_acc": self.train_acc_list,
            #       "val_acc": self.val_acc_list})
                     
            if epochs_stats is not None and set(['val_loss','val_acc', 'train_loss', 'train_acc', 'lr_list']).issubset(epochs_stats.columns):
                loss_dir = os.path.join(self.model_fit_dir, 'loss_curve')
                lossfig_dir = os.path.join(loss_dir, 'figure')
                save_training_process(epochs_stats, plot_dir=lossfig_dir)
            
            # generate the output of the testset

            clogger.critical('>'*40)
            clogger.critical('End fit.')
            pre_lab_all, label_all = self.eval_testset(model, clogger)
            
            tgt_result_file = result_file if result_file is not None else self.model_result_file # add to allow the external configuration
            
            if innerSaving:
                np.savez(tgt_result_file, pred = pre_lab_all, label= label_all)
                clogger.critical('Save result file to the location: {}'.format(tgt_result_file))
            clogger.critical('-'*80)   
        
            return pre_lab_all, label_all, epochs_stats
            
        except:
            clogger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def eval_testset(self, model, logger):
        if logger is not None:
            logger.critical('>'*40)
            logger.critical('Evaluation on the testing set.')
            
        with torch.no_grad():
            
            if self.model_opts.arch == 'torch_nn':
                model.eval()
                       
            test_sample_list, test_lable_list = self.data_opts.load_testset()

            pre_lab_all = []
            label_all = []
            
            for (Sample, Label) in tqdm(zip(test_sample_list, test_lable_list), total=len(test_sample_list)):
                pred_i = []
                label_i = []
                for (sample, label) in zip(Sample, Label):
                    pre_lab = model.predict(sample)
                    pred_i.append(pre_lab)
                    label_i.append(label)
                pred_i = np.concatenate(pred_i)
                label_i = np.concatenate(label_i)
                
                pre_lab_all.append(pred_i)
                label_all.append(label_i)
                
        return pre_lab_all, label_all

    def get_complexity(self, clogger = None):
        self.model_eval_dir = os.path.join(self.eval_dir, self.model_name)
        os_makedirs(self.model_pred_dir)
        if clogger is None:
                clogger = self.logger_config(
                    self.model_eval_dir, 'train')
        
        
        cid_hyper = Opt(self.model_opts.hyper)
        cid_hyper.num_classes = self.data_opts.num_classes
        cid_hyper.model_fit_dir = self.model_fit_dir
        cid_hyper.model_name = self.model_name
        
        model = self.model_import()
        model = model(cid_hyper, clogger) #todo: check model_fit_dir in model and model trainer
        clogger.critical('Loading Model.')
        clogger.critical(f'Model: \n{str(model)}')

        if self.model_opts.arch == 'torch_nn':
            # clogger.info(">>> Total params: {:.2f}M".format(
            #     sum(p.numel() for p in list(model.parameters())) / 1000000.0))
            from thop import profile
            dummy_input = torch.randn(1, 2, 128).to(next(model.parameters()).device)
            flops, params = profile(model, (dummy_input,))
            # print('the flops is {}G, the params is {}M'.format(round(flops / (10 ** 9), 2), round(params / (10 ** 6), 2)))  
            clogger.info(">>> Total flops: {:.2f}M Total params: {:.2f}M  ".format(round(flops / (10 ** 6), 2), round(params / (10 ** 6), 2))) 
    
    def get_inference_time(self, clogger = None):
        self.model_eval_dir = os.path.join(self.eval_dir, self.model_name)
        os_makedirs(self.model_pred_dir)
        if clogger is None:
                clogger = self.logger_config(
                    self.model_eval_dir, 'train')
                
        cid_hyper = Opt(self.model_opts.hyper)
        cid_hyper.num_classes = self.data_opts.num_classes
        cid_hyper.model_fit_dir = self.model_fit_dir
        cid_hyper.model_name = self.model_name
        
        model = self.model_import()
        model = model(cid_hyper, clogger) #todo: check model_fit_dir in model and model trainer
        clogger.critical('Loading Model.')
        clogger.critical(f'Model: \n{str(model)}')


        dummy_input = torch.randn(1, 2, 128).to(next(model.parameters()).device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings=np.zeros((repetitions,1))
        #GPU-WARM-UP
        for _ in range(10):
            _ = model(dummy_input)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        mean_fps = 1000. / mean_syn
        print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
        print(mean_syn)
        clogger.info(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
        clogger.info(mean_syn)


    def evaluate(self, elogger = None, force_update=False, ave_confMax = False):
        eLogger = set_logger(os.path.join(self.eval_dir, '{}.{}.eval.log'.format(self.data_name, self.model_name)), '{}.{}'.format(
                self.data_name, self.model_name.upper()), self.logger_level) if elogger is None else elogger
        
        if self.data_statue is False:
            self.load_data(logger=eLogger)
        
        self.model_eval_dir = os.path.join(self.eval_dir, self.model_name)
        self.eval_acc_dir = os.path.join(self.model_eval_dir, 'accuracy')
        self.eval_plot_dir = os.path.join(self.model_eval_dir, 'figures')
        
        os_makedirs(self.eval_acc_dir)
        os_makedirs(self.eval_plot_dir)
        os_makedirs(self.model_pred_dir)
        
        # for i in self.cid_list: # multiple cross validation in the future version
        
        if self.fit_statue:
            force_update = False
        
        if os.path.exists(self.model_result_file) and force_update is False:
            with np.load(self.model_result_file) as data:
                pre_lab_all, label_all = data['pred'], data['label']
        else:
            pre_lab_all, label_all,_ = self.conduct_fit()
        
        
        Confmat_Set = np.zeros((len(self.data_opts.num_snrs), self.data_opts.num_classes, self.data_opts.num_classes), dtype=int)
        Accuracy_list = np.zeros(len(self.data_opts.num_snrs), dtype=float)
        
        for snr_i, (pred_i, label_i) in enumerate(zip(pre_lab_all, label_all)):
            cm_i =  confusion_matrix(label_i, pred_i)
            Confmat_Set[snr_i, :, :] = cm_i
            Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)
    
        pre_lab_all = np.concatenate(pre_lab_all)
        label_all = np.concatenate(label_all)
        
        F1_score = f1_score(label_all, pre_lab_all, average='macro')
        kappa = cohen_kappa_score(label_all, pre_lab_all)
        acc = np.mean(Accuracy_list)
        
        eLogger.info('Overall Accuracy is: {:.3f}%'.format(acc * 100))
        eLogger.info(f'Macro F1-score is: {F1_score}')
        eLogger.info(f'Kappa Coefficient is: {kappa}')
        
        if ave_confMax:
            save_confmat(Confmat_Set, self.data_opts.num_snrs, self.data_opts.classes, self.eval_plot_dir)
        
        
        Accuracy_Mods = save_snr_acc(Accuracy_list, Confmat_Set, self.data_opts.num_snrs, self.data_name, self.data_opts.classes.keys(), self.eval_plot_dir)
        
        tgt_acc_file = os.path.join(self.eval_acc_dir, 'acc.npz')
        np.savez(tgt_acc_file, acc_overall = Accuracy_list, acc_mods= Accuracy_Mods)
        eLogger.info('Save accuracy file to the location: {}'.format(tgt_acc_file))
        
        return F1_score, kappa, acc

    def tuning(self):
        try:
            self.tune = True
            self.best_checkpoint_path = None
            
            if 'innerTuning' in self.model_opts.dict:
                if self.model_opts.innerTuning == True:
                    self.tune = False
            
            if self.tune:
                tuner_dir = os.path.join(self.model_fit_dir, 'tuner')
                os_makedirs(tuner_dir)
                
                self.model_opts.tuner.dir = tuner_dir
                tLogger = self.logger_config(tuner_dir, 'tuning')
                
                pT_hyper = Opt()            
                if 'preTuning_model_path' in self.model_opts.tuner.dict:
                    pT_path = self.model_opts.tuner.preTuning_model_path
                    if os.path.exists(pT_path):
                        pT_hyper.merge(torch.load(pT_path))
                        self.model_opts.hyper.update(pT_hyper)
                        
                        tLogger.critical('-'*80)
                        for (arg, value) in pT_hyper.dict.items():
                            tLogger.info("PreTuning Results:\t %s - %r", arg, value)
                    else:
                        raise ValueError('Non-found preTuning results: {}.\nPlease check the preTuning_model: {}'.format(pT_path, self.model_opts.tuner.preTuning_model))
                # else:
                if len(list(self.model_opts.tuning.dict.keys())) > 0:
                    
                    if self.data_statue is False:
                        self.load_data(logger=tLogger)
                    
                    series_tuner = HyperTuner(
                        self.model_opts, tLogger, self.data_opts)
                    best_hyper, best_checkpoint_path = series_tuner.conduct() # best_hyper is an Obj
                else:
                    best_hyper = Opt()
                
                
                pT_hyper.merge(best_hyper)
                self.best_hyper = pT_hyper
                self.best_checkpoint_path = best_checkpoint_path
                
                tLogger.critical('-'*80)
                tLogger.critical('Tuning complete.')
                
                if isinstance(pT_hyper, Mapping):
                    pT_hyper_info = pT_hyper
                elif isinstance(pT_hyper, object):
                    pT_hyper_info = vars(pT_hyper)
                else:
                    raise ValueError('Error data type in pT_hyper from: {}'.format(tuner_dir))
                                        
                for (arg, value) in pT_hyper_info.items():
                    tLogger.info("Tuning Results:\t %s - %r", arg, value)
                    
            return self.tune               
        except:
            self.tune = False
            tLogger.exception(
                '{}\nGot an error on tuning.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    # def load_tuning(self):
    #     tuner_dir = os.path.join(self.model_fit_dir, 'tuner')
    #     tuner_path = os.path.join(tuner_dir, 'hyperTuning.best.pt')
    #     best_hyper = torch.load(tuner_path)
    #     if not os.path.exists(tuner_path):
    #         raise ValueError(
    #             'Invalid tuner path: {}'.format(tuner_path))
    #     return best_hyper

if __name__ == "__main__":
    args = get_parser()
    args.cuda = True
    
    args.exp_config = 'exp_config'
    args.exp_file= 'rml16a'
    # args.exp_name = 'icassp23'
    args.exp_name = 'tuning.mcl'
    args.gid = 0
    
    args.test = True
    args.clean = True
    args.model = 'mcl2'
    
    
    task = Task(args)
    task.tuning()
    task.conduct()
    # task.evaluate(force_update=True)     