CUDA_VISIBLE_DEVICES=1 python main.py --source mnist --target usps --nepoch 20 --milestone_1 10 --milestone_2 15 --outf output/mnist_usps
CUDA_VISIBLE_DEVICES=1 python main.py --source mnist --target usps --nepoch 20 --milestone_1 10 --milestone_2 15 --rotation --outf output/mnist_usps_r
# CUDA_VISIBLE_DEVICES=1 python main.py --source mnist --target usps --nepoch 20 --milestone_1 10 --milestone_2 15 --rotation --quadrant --outf output/mnist_usps_rq
