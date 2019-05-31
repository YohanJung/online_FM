#!/usr/bin/env bash
#!bin/sh
# dos2unix [filname]

#setup_model_list = ['FM_FTRL','SFTRL_C','SFTRL_V','RFF','All']
#setup_dataset_list = ["movie100k","codrna","Frappe","YEARMSD"]

# for test
#for iter_model in FM_FTRL SFTRL_C SFTRL_V
#do
#python3 test_argparse.py --num_exp=1 --dataset=codrna --model=${iter_model} --task=reg --lr_FM=0.01 &
#done

# nohup test
#nohup python3 test_argparse.py --num_exp=1 --dataset=codrna --model=All --task=reg --lr_FM=0.01&
#nohup python3 test_argparse.py --num_exp=1 --dataset=codrna --model=All --task=reg --lr_FM=0.01 1>/dev/null/ 2>&1

####################################################################################################################
# FM-reg

# over learning rate
#python3 test_argparse.py --num_exp=1 --dataset=YEARMSD --model=FM_FTRL --task=reg --FM_feature_dim_list 5 --lr_FM_list 0.0001 0.0005 0.001 0.005 0.01 0.05 &
#python3 test_argparse.py --num_exp=1 --dataset=YEARMSD --model=SFTRL_C --task=reg --FM_feature_dim_list 5 --lr_FM_list 0.0001 0.0005 0.001 0.005 0.01 0.05 &
#python3 test_argparse.py --num_exp=1 --dataset=YEARMSD --model=SFTRL_V --task=reg --FM_feature_dim_list 5 --lr_FM_list 0.0001 0.0005 0.001 0.005 0.01 0.05 &
#python3 test_argparse.py --num_exp=1 --dataset=movie100k --model=SFTRL_V --task=reg --FM_feature_dim_list 5 --lr_FM_list 0.0001 0.0005 0.001 0.005 0.01 0.05 &

# over feature dim list
#python3 test_argparse.py --num_exp=1 --dataset=YEARMSD --model=FM_FTRL --task=reg --FM_feature_dim_list 5 10 20 30 50 --lr_FM_list 0.001 &
#python3 test_argparse.py --num_exp=1 --dataset=YEARMSD --model=SFTRL_C --task=reg --FM_feature_dim_list 5 10 20 30 50 --lr_FM_list 0.001 &
#python3 test_argparse.py --num_exp=1 --dataset=YEARMSD --model=SFTRL_V --task=reg --FM_feature_dim_list 5 10 20 30 50 --lr_FM_list 0.001 &


# RRF-reg

# over movielens 100k RRF
#python3 test_argparse.py --num_exp=1 --dataset=movie100k --model=RRF --purpose=sep --task=reg --RRF_gamma_list 0.01 0.1 1  --lr_RRF_w_list 0.001 0.005 0.01 0.05 --lr_RRF_gamma_list 0.001 0.005 0.01 0.05 --RRF_num_sample_spectral_list 5 10 20 &

#python3 test_argparse.py --num_exp=1 --dataset=YEARMSD --model=RRF --purpose=sep --task=reg --RRF_gamma_list 0.01 0.1 1  --lr_RRF_w_list 0.001 0.005 0.01 0.05 --lr_RRF_gamma_list 0.001 0.005 0.01 0.05 --RRF_num_sample_spectral_list 5 10 20 &

# over YEARMSD RND100k RRF parametrization
#python3 test_argparse.py --num_exp=1 --dataset=YEARMSD --model=RRF --task=reg --RRF_gamma_list 0.01 0.1  --lr_RRF_w_list 0.0001 0.001 0.01 --lr_RRF_gamma_list 0.000001 --RRF_num_sample_spectral_list 10 20 30 50 &



####################################################################################################################











# over codrna
#python3 test_argparse.py --num_exp=1 --dataset=codrna --model=RRF --task=cls --RRF_gamma_list 0.01 0.1 1. --lr_RRF_w_list 0.001 0.005 0.01 --lr_RRF_gamma_list 0.001 0.005 0.01 --RRF_num_sample_spectral_list 10 20 30 50&

#python3 test_argparse.py --num_exp=1 --dataset=codrna --model=FM_FTRL --task=cls --FM_feature_dim_list 10 --lr_FM_list 0.0001 0.0005 0.001 0.005 0.01 0.05 &
python3 test_argparse.py --num_exp=1 --dataset=codrna --model=FM_FTRL --task=cls --FM_feature_dim_list 5 10 20 30 50 --lr_FM_list 0.05 &

#python3 test_argparse.py --num_exp=1 --dataset=codrna --model=SFTRL_C --task=cls --FM_feature_dim_list 10 --lr_FM_list 0.0001 0.0005 0.001 0.005 0.01 0.05 &
#python3 test_argparse.py --num_exp=1 --dataset=codrna --model=SFTRL_C --task=cls --FM_feature_dim_list 10 20 30 40 50  --lr_FM_list 0.05 &
#python3 test_argparse.py --num_exp=1 --dataset=codrna --model=SFTRL_V --task=cls --FM_feature_dim_list 10 20 30 40 50  --lr_FM_list 0.05 &
#python3 test_argparse.py --num_exp=1 --dataset=codrna --model=SFTRL_C --task=cls --FM_feature_dim_list 5 10 20 30 50 --lr_FM_list 0.0005 &


