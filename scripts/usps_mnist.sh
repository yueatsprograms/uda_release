CUDA_VISIBLE_DEVICES=1 python main.py --width 4 --nepoch 40 --milestone_1 30 --milestone_2 35 --source usps --target mnist --rotation --outf output/usps_mnist
CUDA_VISIBLE_DEVICES=1 python main.py --width 4 --nepoch 40 --milestone_1 30 --milestone_2 35 --source usps --target mnist --rotation --outf output/usps_mnist_r
# CUDA_VISIBLE_DEVICES=1 python main.py --width 4 --nepoch 40 --milestone_1 30 --milestone_2 35 --source usps --target mnist --rotation --quadrant --outf output/usps_mnist_rq
