#!/bin/bash

./multilauncher.sh 0 KITTI SegNetModel dSegNetModel DispSegNetModel DispSegNetBasicModel EncFuseModel PydSegNetModel
./multilauncher.sh 0 CityScapes SegNetModel dSegNetModel DispSegNetModel DispSegNetBasicModel EncFuseModel PydSegNetModel
# ./multilauncher.sh 0 BDD10k SegNetModel dSegNetModel DispSegNetModel DispSegNetBasicModel EncFuseModel PydSegNetModel