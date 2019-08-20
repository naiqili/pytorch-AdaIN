# export TMPDIR="/home/chengbin/naiqi/pytorch-AdaIN/logs"

CUDA_VISIBLE_DEVICES=3,6,7 python train.py --content_dir "/data/chengbin/ILSVRC2012/train/" --style_dir "/data/chengbin/style/train/" --vgg "../stylize-datasets/models/vgg_normalised.pth" --log_dir "/home/chengbin/naiqi/pytorch-AdaIN/logs"