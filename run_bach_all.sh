#!/usr/bin/env bash
#!bin/sh
# dos2unix [filname]



#setup_model_list = ['FM_FTRL','SFTRL_C','SFTRL_V','RFF','All']
#setup_dataset_list = ["movie100k","codrna","Frappe","YEARMSD"]



# over movielens100k feature dim list

#python3 test_argparse.py --num_exp=1 --dataset=movie100k --model_list SFTRL_V SFTRL_C FM_FTRL RRF --task=reg --purpose=all&


# over YEARMSD learning rate

#python3 test_argparse.py --num_exp=1 --dataset=YEARMSD --model_list SFTRL_V SFTRL_C FM_FTRL RRF --task=reg --purpose=all&


# over codrna learning rate

python3 test_argparse.py --num_exp=1 --dataset=codrna --model_list SFTRL_V SFTRL_C FM_FTRL RRF --task=cls --purpose=all &


# over Frappe learning rate

#python3 test_argparse.py --num_exp=1 --dataset=Frappe --model_list SFTRL_V SFTRL_C FM_FTRL RRF --task=cls --purpose=all &
