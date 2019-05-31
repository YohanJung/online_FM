import os
import argparse
import logging

import tracemalloc
import torch
import numpy as np
import time

from data_manager import *
from models.SFTRL_CCFM import SFTRL_CCFM
from models.SFTRL_Vanila import SFTRL_Vanila
from models.FM_FTRL import  FM_FTRL
from models.RRF_Online import RRF_Online
#from models.RRF import RRF

#from peformance_manager import *
from data_manager import *

Tensor_Type = torch.DoubleTensor


parser = argparse.ArgumentParser(description = 'Online FM Experiment')

#setup_model_list = ['FM_FTRL','SFTRL_C','SFTRL_V','RFF']
setup_model_list = ['FM_FTRL','SFTRL_C','SFTRL_V']
setup_dataset_list = ["movie100k","codrna","Frappe","YEARMSD"]


# num_exp

parser.add_argument('--random_seed', type = int ,default= 1000 ,help = 'random_seed' )
parser.add_argument('--num_exp',type = int, default = 10, help = 'model chose')
parser.add_argument('--task',type = str, default = 'reg', help = 'task selection',
                    choices = ["reg",'cls'] )
parser.add_argument('--purpose',type = str, default = 'sep', choices = ["sep",'all'] , help = 'task selection')


# experiment parameter
parser.add_argument('--FM_feature_dim',type = int, default= 5  ,help = 'FM feature')
parser.add_argument('--FM_feature_dim_list',type = int, nargs= '+',help = 'FM feature list')

#parser.add_argument('--down_sampling',type = int, default= 1  ,help = 'task selection')

# model setup
# parser.add_argument('--model',type = str, default= 'RRF' , help = 'model ready',
#                     choices = setup_model_list)

parser.add_argument('--model',type = str, default= 'RRF' , help = 'model ready')
parser.add_argument('--model_list',nargs= '+', type = str , help = 'model list ready')


# dataset up
parser.add_argument('--dataset',type = str,default= 'movie100k', help = 'dataset ready',
                    choices = setup_dataset_list)

# model learning_rate setup
parser.add_argument('--lr_FM', type = float, default= 0.05 , help = 'learning rate FM')
parser.add_argument('--lr_FM_list', nargs='+', type = float , help = 'learning rate FM list')

parser.add_argument('--lr_RRF',type = float, default= 0.05 , help = 'learning rate RRF')
parser.add_argument('--lr_RRF_w_list',nargs='+', type = float , help = 'learing rate RRF w list')
parser.add_argument('--lr_RRF_gamma_list',nargs='+', type = float , help = 'learing rate RRF gamma list')
parser.add_argument('--RRF_gamma_list',nargs='+', type = float , help = 'RRF gamma list')
parser.add_argument('--RRF_num_sample_spectral_list',nargs='+' , type =int , help = 'RRF number of spectral list')
parser.add_argument('--RRF_loss_type',default = None,help = 'RRF loss type')
parser.add_argument('--loss_type',type = str, default= None, help = 'learning rate RRF')


# save the experiment
#parser.add_argument('--save_model',action = 'store_true')
parser.add_argument('--save_path',default='results/', help = 'save path')
parser.add_argument('--fig_path',default='fig/', help = 'fig save path')
parser.add_argument('--log_path',default='log/', help = 'log_path')


args = parser.parse_args()
args.save_dir = args.save_path + "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}/".format( str(args.dataset),
                                                                             str(args.purpose),
                                                                             str(args.model_list),
                                                                             str(args.num_exp),
                                                                             str(args.task),
                                                                             str(args.lr_FM_list),
                                                                             str(args.FM_feature_dim_list),
                                                                             str(args.RRF_gamma_list),
                                                                             str(args.lr_RRF_w_list),
                                                                             str(args.lr_RRF_gamma_list),
                                                                             str(args.RRF_num_sample_spectral_list),
                                                                             str(args.RRF_loss_type) )
os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else 1


args.model_save_dir = args.save_dir + "model/"
args.fig_save_dir = args.save_dir + "fig/"
os.makedirs(args.model_save_dir) if not os.path.exists(args.model_save_dir) else 1
os.makedirs(args.fig_save_dir) if not os.path.exists(args.fig_save_dir) else 1


args.log_dir = args.save_dir + "log/"
os.makedirs(args.log_dir) if not os.path.exists(args.log_dir) else 1
path_log = args.log_dir + "logger_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11},txt".format( str(args.dataset),
                                                                                          str(args.purpose),
                                                                                          str(args.model_list),
                                                                                          str(args.num_exp),
                                                                                          str(args.task),
                                                                                          str(args.lr_FM_list),
                                                                                          str(args.FM_feature_dim_list),
                                                                                          str(args.RRF_gamma_list),
                                                                                          str(args.lr_RRF_w_list),
                                                                                          str(args.lr_RRF_gamma_list),
                                                                                          str(args.RRF_num_sample_spectral_list),
                                                                                          str(args.RRF_loss_type) )
logger = logging.getLogger('Result_log')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(path_log)
logger.addHandler(file_handler)

logger.info("=="*20)
logger.info("experiment stat")
print("=="*20)

# gpu usage commnad
    # args.cuda = torch.cuda.is_available()
    # np.random.seed(args.random_seed)
    # random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # torch.cuda.manual_seed_all(args.random_seed)


# # model setup

def _load_collection_data(filename,down_sampling = 1,want_permute = False):
    if filename in setup_dataset_list:
        # does not working yet
        if filename  == "movie100k":
            nbUsers,nbMovies,nbRatingsTrain,nbRatingsTest = 943,1682,90570,9430
            nbFeatures = nbUsers + nbMovies
            data_dir = './data/ml-100k/'

            # filename1, filename2 = 'ub.base', './ub.test'
            filename1, filename2 = 'ua.base', './ua.test'
            _, x_train, _ , y_train, timestamp_train = load_dataset_movielens(data_dir + filename1,
                                                                              nbRatingsTrain,
                                                                              nbFeatures,
                                                                              nbUsers)
            x_train_s, y_train_s, _ = sort_dataset_movielens(x_train, y_train, timestamp_train)
            x_train_s = x_train_s.todense()

        # working checked
        elif filename  ==  "codrna" :
            data_dir = './data/cod-rna2/'
            filename = "cod-rna2.scale"
            x_train_s, y_train_s = load_svmlight_file(data_dir + filename, n_features=8)
            x_train_s = x_train_s.toarray()
            #y_train_s = np.asarray([1.0 if i_th > 0 else 0.0 for i_th in y_train_s])

        elif filename  ==  "Frappe" :
            filename = './data/frappe/frappe.train.libfm'
            x_train_s, y_train_s = load_dataset_fappe(filename, logloss_opt=False)

        # working checked
        elif filename  ==  "YEARMSD" :
            filename = './data/YearPredictionMSD/YearPredictionMSD'
            #filename = './data/YearPredictionMSD/YearPredictionMSD_test'
            x_train_s, y_train_s = load_dataset_YearPredictionMSD(filename)
        else :
            return
    else :
        raise ValueError('NOT found dataset')
    if want_permute:
        idx = np.random.permutation(x_train_s.shape[0])
        x_train_s = x_train_s[idx]
        y_train_s = y_train_s[idx]
    else:
        pass

    return  Tensor_Type(x_train_s), Tensor_Type(y_train_s)



def evaluation_metric(pred,real,task):
    if task == 'reg':
        val = 0.0
        metric = [np.inf]
        for idx, (pred_idx, real_idx) in enumerate(zip(pred, real)):
            val += (pred_idx - real_idx) ** 2
            metric.append((1 / (idx + 1)) * val)
        metric = np.asarray(metric).reshape([-1, 1])
        return metric
    elif task == 'cls':
        val = 0.0
        metric = [np.inf]
        metric_cnt = [np.inf]
        for idx, (pred_idx, real_idx) in enumerate(zip(pred, real)):
            val += 1 if pred_idx == real_idx else 0
            metric.append(1 / (idx + 1) * np.log(1.0 + np.exp(-pred_idx * real_idx)))
            metric_cnt.append(1 / (idx + 1) * val)
        metric = np.asarray(metric).reshape([-1, 1])
        metric_cnt = np.asarray(metric_cnt).reshape([-1, 1])

        return metric, metric_cnt
    else :
        raise NotImplementedError('task does not matched')
    return


def _make_FM_model(x_train_s,y_train_s,model_name,task,lr_FM,feature_dim):

    #task, learning_rate, num_feature
    if model_name  == 'FM_FTRL':
        model = FM_FTRL(x_train_s, y_train_s, task, lr_FM, feature_dim)
    elif model_name  == 'SFTRL_C':
        model = SFTRL_CCFM(x_train_s, y_train_s, task, lr_FM, feature_dim)
    elif model_name == 'SFTRL_V':
        model = SFTRL_Vanila(x_train_s, y_train_s, task, lr_FM, feature_dim)
    else:
        return
    return model

def _make_RRF_model(x_train_s,y_train_s,
                    task,
                    loss_type,
                    gamma,w,
                    num_sampled_spectral,lr_RRF_w,lr_RRF_gamma ):

    #task, learning_rate, num_feature
    model = RRF_Online(x_train_s, y_train_s,
                       task,
                       loss_type=loss_type,
                       gamma=gamma,
                       w=w,
                       num_sampled_spectral=num_sampled_spectral,
                       random_seed=100,
                       lr_RRF_w=lr_RRF_w,
                       lr_RRF_gamma=lr_RRF_gamma)

    return model



def _run_exp_single_model(args,model,x_train_s,y_train_s):
    print('chosen model {}'.format(str(model)))
    result_pred_list = {}
    result_time_list = {}
    tracemalloc.start(10)
    time1 = tracemalloc.take_snapshot()

    if model != 'RRF':
        for i_th_lr in args.lr_FM_list :
            for i_th_feature_dim in args.FM_feature_dim_list:
                current_model = _make_FM_model(x_train_s,y_train_s, model, args.task, i_th_lr, i_th_feature_dim )
                temp_name = model + '_' + str(i_th_lr) + '_' + str(i_th_feature_dim)
                result_pred_list[temp_name + '_pred'], _, result_time_list[temp_name + '_time'] = current_model.online_learning(logger)

                # try:
                #     result_pred_list[temp_name + '_pred'], _, result_time_list[temp_name + '_time'] = current_model.online_learning(logger)
                # except:
                #     msg = temp_name + ' get error'
                #     logger.info(msg)
                #     print(msg)
                #     pass

    elif model == 'RRF':
        for i_th_num_sample in args.RRF_num_sample_spectral_list:
            for i_th_Gamma in args.RRF_gamma_list:
                for i_th_RRF_lr_w in args.lr_RRF_w_list:
                    for i_th_RRF_lr_gamma in args.lr_RRF_gamma_list:

                        current_model = _make_RRF_model(x_train_s,y_train_s,
                                                        args.task,args.loss_type,
                                                        gamma = i_th_Gamma, w = None,
                                                        num_sampled_spectral=i_th_num_sample,
                                                        lr_RRF_w=i_th_RRF_lr_w,
                                                        lr_RRF_gamma=i_th_RRF_lr_gamma )
                        temp_name = model + '_' + str(i_th_num_sample) + '_' + str(i_th_Gamma) + '_' + str(i_th_RRF_lr_w) + '_'+ str(i_th_RRF_lr_gamma)

                        # result_pred_list[temp_name + '_pred'], _, result_time_list[temp_name + '_time'] = current_model.online_learning(logger)

                        try:
                            result_pred_list[temp_name + '_pred'], _, result_time_list[temp_name + '_time'] = current_model.online_learning(logger)
                        except:
                            msg = temp_name + ' get error'
                            logger.info(msg)
                            print(msg)
                            pass
    else :
        return
    time2 = tracemalloc.take_snapshot()
    stats = time2.compare_to(time1,'lineno')
    logger.info('')
    logger.info('memory management')
    for stat in stats[:5]:
        print(stat)
        logger.info(stat)
    result_pred_list['real'] = y_train_s
    return  result_pred_list,result_time_list


def _conduct_exp_multiple_model(model_list,x_train_s,y_train_s):
    result_pred_list = {}
    result_time_list = {}

    for model_i_th in model_list:

        print(model_i_th + '_loaded')
        if model_i_th != 'RRF':
            if model_i_th == 'FM_FTRL':
                #i_th_lr, i_th_feature_dim = 0.005,5 # movielens
                #i_th_lr, i_th_feature_dim = 0.01, 5 #YEARMSD
                i_th_lr, i_th_feature_dim = 0.05,5 #cod-rna
                #i_th_lr, i_th_feature_dim = 0.005,5 # frappe

            else :
                #i_th_lr, i_th_feature_dim = 0.05,5 # movielens
                #i_th_lr, i_th_feature_dim = 0.001, 5 #YEARMSD
                i_th_lr, i_th_feature_dim = 0.05,5 # cod-rna
                #i_th_lr, i_th_feature_dim = 0.005,5 # frappe


            current_model = _make_FM_model(x_train_s, y_train_s, model_i_th, args.task, i_th_lr, i_th_feature_dim)
            temp_name = model_i_th + '_' + str(i_th_lr) + '_' + str(i_th_feature_dim)
            #result_pred_list[temp_name + '_pred'], _, result_time_list[temp_name + '_time'] = current_model.online_learning(logger)

            # try:
            #     result_pred_list[temp_name + '_pred'], _, result_time_list[temp_name + '_time'] = current_model.online_learning(logger)
            # except:
            #     msg = temp_name + ' get error'
            #     logger.info(msg)
            #     print(msg)
            #     pass

        elif model_i_th == 'RRF':

            #i_th_num_sample, i_th_Gamma , i_th_RRF_lr_w ,i_th_RRF_lr_gamma = 10,.1, 0.01,0.01 # movielens
            #i_th_num_sample, i_th_Gamma, i_th_RRF_lr_w, i_th_RRF_lr_gamma = 50, .1, 0.001, 1e-06 # YEARMSD
            i_th_num_sample, i_th_Gamma, i_th_RRF_lr_w, i_th_RRF_lr_gamma = 5, .1, 0.001, 0.001 # cod-rna
            #i_th_num_sample, i_th_Gamma, i_th_RRF_lr_w, i_th_RRF_lr_gamma = 10, .1, 0.001, 0.001 # cod-rna

            current_model = _make_RRF_model(x_train_s, y_train_s,
                                            args.task, args.loss_type,
                                            gamma=i_th_Gamma, w=None,
                                            num_sampled_spectral=i_th_num_sample,
                                            lr_RRF_w=i_th_RRF_lr_w,
                                            lr_RRF_gamma=i_th_RRF_lr_gamma)
            temp_name = model_i_th + '_' + str(i_th_num_sample) + '_' + str(i_th_Gamma) + '_' + str(i_th_RRF_lr_w) + '_' + str(i_th_RRF_lr_gamma)

            #result_pred_list[temp_name + '_pred'], _, result_time_list[temp_name + '_time'] = current_model.online_learning(logger)
            # try:
            #     result_pred_list[temp_name + '_pred'], _, result_time_list[
            #         temp_name + '_time'] = current_model.online_learning(logger)
            # except:
            #     msg = temp_name + ' get error'
            #     logger.info(msg)
            #     print(msg)
            #     pass
        else:
            pass


        #result_pred_list[temp_name + '_pred'], _, result_time_list[temp_name + '_time'] = current_model.online_learning(logger)
        try:
            result_pred_list[temp_name + '_pred'], _, result_time_list[
                temp_name + '_time'] = current_model.online_learning(logger)
        except:
            msg = temp_name + ' get error'
            logger.info(msg)
            print(msg)
            pass

    result_pred_list['real'] = y_train_s
    return result_pred_list,result_time_list



def _result_experiment(result_dict,result_time_dict,args,single_model_name):
     #prediction task figure
     fig = plt.figure(figsize = (8,4))
     for i_th_result_dict in result_dict:

         #print(result_dict[i_th_result_dict])
         if torch.is_tensor(result_dict[i_th_result_dict]):
             plt.plot(result_dict[i_th_result_dict].data.numpy()  ,'.' , label = i_th_result_dict ,alpha = 0.5)
         else :
             plt.plot(result_dict[i_th_result_dict] , '.', label=i_th_result_dict ,alpha = 0.5)

     #plt.legend(bbox_to_anchor=(1,0) , loc='lower center' ,  bbox_transform=fig.transFigure , ncol = len(result_dict))
     plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.1) , bbox_transform=fig.transFigure, ncol = 3  )
     plt.title(args.dataset + '_' + args.task)

     #plt.ylim([0,6.0])

     plt.show()
     print(args.fig_save_dir + single_model_name + '_' + 'learning_rate_comparison' +'.png' + ' is saved')
     plt.savefig(args.fig_save_dir  + single_model_name + '_' + 'learning_rate_comparison' +'.png', bbox_inches="tight")


    # performance task figure
     fig = plt.figure(figsize = (8,4))
     if args.task == "cls" :
         for i_th_result_dict in result_dict:
             if i_th_result_dict is not 'real':
                 _,metric_cnt = evaluation_metric(result_dict[i_th_result_dict] , result_dict['real'] , args.task)
                 #plt.plot(metric_cnt , label=i_th_result_dict, alpha=0.5)
                 plt.plot(np.log(1 + metric_cnt), label=i_th_result_dict, alpha=0.5)
                 plt.ylabel('metric')
     elif args.task == "reg":
         for i_th_result_dict in result_dict:
             if i_th_result_dict is not 'real':
                 metric = evaluation_metric(result_dict[i_th_result_dict] , result_dict['real'] , args.task)
                 plt.plot(np.log(1 + metric), label=i_th_result_dict, alpha=0.5)
                 plt.ylabel('log(1 + metric) ')
     else :
         raise ValueError('not implemented')

     plt.xlabel('iteration')

     plt.title(args.dataset + '_' + args.task)
     plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.1) , bbox_transform=fig.transFigure, ncol= 3  )
     plt.show()
     print(args.fig_save_dir + single_model_name + '_' +'metric_over' +'.png'  + ' is saved')
     plt.savefig(args.fig_save_dir  +  single_model_name + '_' + 'metric_over' +'.png', bbox_inches="tight")


    # training_time
     fig = plt.figure(figsize = (8,4))
     plt.bar(result_time_dict.keys(),result_time_dict.values(), align = 'center', alpha=0.5)
     plt.xticks(np.arange(len(result_time_dict.keys())) , result_time_dict.keys() ,rotation = 90)
     plt.ylabel('seconds ')
     plt.title(args.dataset + '_' + args.task + '_time')
     plt.show()
     print(args.fig_save_dir + single_model_name + '_' +'time_over' +'.png'  + ' is saved')
     plt.savefig(args.fig_save_dir  +  single_model_name + '_' + 'time_over' +'.png', bbox_inches="tight")






if __name__ == "__main__":
    x_train_s, y_train_s = _load_collection_data(args.dataset , down_sampling = 1, want_permute = False)

    if args.purpose == 'sep':
        result_pred_dict,result_time_dict = _run_exp_single_model(args,args.model,x_train_s,y_train_s)
        _result_experiment(result_pred_dict, result_time_dict, args, args.model)
    elif args.purpose == 'all':
        result_pred_dict,result_time_dict = _conduct_exp_multiple_model(args.model_list, x_train_s, y_train_s)
        _result_experiment(result_pred_dict, result_time_dict, args, args.purpose)
    else :
        pass






   # # prediction task figure
   #  fig = plt.figure(figsize = (8,4))
   #  for i_th_result_dict in result_dict:
   #      #print(result_dict[i_th_result_dict])
   #      if torch.is_tensor(result_dict[i_th_result_dict]):
   #          plt.plot(result_dict[i_th_result_dict].data.numpy()  ,'.' , label = i_th_result_dict ,alpha = 0.5)
   #      else :
   #          plt.plot(result_dict[i_th_result_dict] , '.', label=i_th_result_dict ,alpha = 0.5)
   #
   #  #plt.legend(loc = 'best')
   #  #plt.legend(bbox_to_anchor=(1,0) , loc='lower right' ,  bbox_transform=fig.transFigure , ncol = 2)
   #  #plt.legend(bbox_to_anchor=(1,0) , loc='lower center' ,  bbox_transform=fig.transFigure , ncol = len(result_dict))
   #  plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.05) , bbox_transform=fig.transFigure, ncol= int(len(result_dict)/2) , )
   #  plt.show()
   #  print(args.fig_save_dir + args.model + 'learning_rate_comparison' +'.png')
   #  plt.savefig(args.fig_save_dir + args.model + 'learning_rate_comparison' +'.png', bbox_inches="tight")
   #
   #
   #  fig = plt.figure(figsize = (8,4))
   #  # performance task figure
   #  if args.task == "cls" :
   #      for i_th_result_dict in result_dict:
   #          if i_th_result_dict is not 'real':
   #              _,metric_cnt = classfication_metric(result_dict[i_th_result_dict] , result_dict['real'])
   #              plt.plot(metric_cnt , label=i_th_result_dict, alpha=0.5)
   #  elif args.task == "reg":
   #      for i_th_result_dict in result_dict:
   #          if i_th_result_dict is not 'real':
   #              metric = regression_metric(result_dict[i_th_result_dict] , result_dict['real'])
   #              plt.plot(metric, label=i_th_result_dict, alpha=0.5)
   #
   #  else :
   #      raise ValueError('not implemented')
   #
   #  plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.05) , bbox_transform=fig.transFigure, ncol= int(len(result_dict)/2)  )
   #  plt.show()
   #  print(args.fig_save_dir + args.model +'metric_over_learning_rate' +'.png')
   #  plt.savefig(args.fig_save_dir + args.model + 'metric_over_learning_rate' +'.png', bbox_inches="tight")
   #
