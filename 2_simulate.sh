# export TMPDIR="/home/chengbin/naiqi/pytorch-AdaIN/logs"

CUDA_VISIBLE_DEVICES=5 python sim_train.py --content_dir "/data/chengbin/ILSVRC2012/train/" --vgg "../stylize-datasets/models/vgg_normalised.pth" --log_dir "/home/chengbin/naiqi/pytorch-AdaIN/logs/sw0001/" --style_weight 0.0001