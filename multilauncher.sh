#!/bin/bash

echo "Inputs to sh"
echo '$0 = ' $0 # Program name
echo '$1 = ' $1	# Cuda
echo '$2 = ' $2 # Dataset
echo '$3 = ' $3 # Model1
echo '$4 = ' $4 # Model2
echo '$5 = ' $5 # Model3
echo '$6 = ' $6 # Model4
echo '$7 = ' $7 # Model5
echo '$8 = ' $8 # Model6
echo '$3$2 = ' $3$2 # WeightNameModel1

python36 launcher.py --cuda $1  --dataset $2 --model $3 --epochs 100 --train_batches 1 --task traineval

cp -r /WeightModels/exjobb/$3$2 /WeightModels/exjobb/WorthyWeights/
#python36 launcher.py --cuda $1  --dataset $2 --model $3  --task evaluate
#python36 launcher.py --cuda $1  --dataset $2 --model $3  --task predict
python36 launcher.py --cuda $1  --dataset $2 --model $3 --task dist --bestweights True
############################################
python36 launcher.py --cuda $1 --dataset $2 --model $4 --epochs 100 --train_batches 1 --task traineval

cp -r /WeightModels/exjobb/$4$2 /WeightModels/exjobb/WorthyWeights/
#python36 launcher.py --cuda $1 --dataset $2 --model $4  --task evaluate
#python36 launcher.py --cuda $1 --dataset $2 --model $4  --task predict
python36 launcher.py --cuda $1 --dataset $2 --model $4 --task dist --bestweights True
############################################
python36 launcher.py --cuda $1 --dataset $2 --model $5 --epochs 100 --train_batches 1 --task traineval

cp -r /WeightModels/exjobb/$5$2 /WeightModels/exjobb/WorthyWeights/
#python36 launcher.py --cuda $1 --dataset $2 --model $5  --task evaluate
#python36 launcher.py --cuda $1 --dataset $2 --model $5  --task predict
python36 launcher.py --cuda $1 --dataset $2 --model $5 --task dist --bestweights True
############################################
python36 launcher.py --cuda $1 --dataset $2 --model $6 --epochs 100 --train_batches 1 --task traineval

cp -r /WeightModels/exjobb/$6$2 /WeightModels/exjobb/WorthyWeights/
#python36 launcher.py --cuda $1 --dataset $2 --model $6  --task evaluate
#python36 launcher.py --cuda $1 --dataset $2 --model $6  --task predict
python36 launcher.py --cuda $1 --dataset $2 --model $6 --task dist --bestweights True
############################################
python36 launcher.py --cuda $1 --dataset $2 --model $7 --epochs 100 --train_batches 1 --task traineval

cp -r /WeightModels/exjobb/$7$2 /WeightModels/exjobb/WorthyWeights/
#python36 launcher.py --cuda $1 --dataset $2 --model $7  --task evaluate
#python36 launcher.py --cuda $1 --dataset $2 --model $7  --task predict
python36 launcher.py --cuda $1 --dataset $2 --model $7 --task dist --bestweights True
############################################
python36 launcher.py --cuda $1 --dataset $2 --model $8 --epochs 100 --train_batches 1 --task traineval

cp -r /WeightModels/exjobb/$8$2 /WeightModels/exjobb/WorthyWeights/
#python36 launcher.py --cuda $1 --dataset $2 --model $8  --task evaluate
#python36 launcher.py --cuda $1 --dataset $2 --model $8  --task predict
python36 launcher.py --cuda $1 --dataset $2 --model $8 --task dist --bestweights True


# ./multilauncher.sh 0 CityScapes SegNetModel dSegNetModel DispSegNetModel DispSegNetBasicModel EncFuseModel PydSegNetModel
# ./multilauncher.sh 1 CityScapes dSegNetModel DispSegNetBasicModel EncFuseModel
# ./multilauncher.sh 0 CityScapes SegNetModel dSegNetModel

# ./multilauncher.sh 0 BDD10k DispSegNetBasicModel SegNetModel
# ./multilauncher.sh 1 BDD10k DispSegNetModel dSegNetModel

# ./multilauncher.sh 0 KITTI DispSegNetBasicModel SegNetModel
# ./multilauncher.sh 1 KITTI DispSegNetModel dSegNetModel
# ./multilauncher.sh 0 KITTI DispSegNetBasicModel DispSegNetModel
# ./multilauncher.sh 0 KITTI SegNetModel dSegNetModel DispSegNetModel DispSegNetBasicModel EncFuseModel PydSegNetModel