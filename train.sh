#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset office --source 0 --target 1 --lr 0.01 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset office --source 0 --target 2 --lr 0.01 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset office --source 1 --target 2 --lr 0.01 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset office --source 2 --target 1 --lr 0.01 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset office --source 1 --target 0 --lr 0.01 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset office --source 2 --target 0 --lr 0.01 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset officehome --source 0 --target 1 --lr 0.0001 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset officehome --source 0 --target 2 --lr 0.0001 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset officehome --source 0 --target 3 --lr 0.0001 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset officehome --source 1 --target 0 --lr 0.0001 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset officehome --source 1 --target 2 --lr 0.0001 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset officehome --source 1 --target 3 --lr 0.0001 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset officehome --source 2 --target 0 --lr 0.0001 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset officehome --source 2 --target 1 --lr 0.0001 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset officehome --source 2 --target 3 --lr 0.0001 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset officehome --source 3 --target 0 --lr 0.0001 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset officehome --source 3 --target 1 --lr 0.0001 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset officehome --source 3 --target 2 --lr 0.0001 --KK 5
CUDA_VISIBLE_DEVICES=0 python train_target.py --target_type OPDA --dataset visda --source 0 --target 1 --lr 0.001 --KK 100
