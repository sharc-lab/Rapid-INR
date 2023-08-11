import os
import subprocess as sp
import shlex

mini_label = f"/export/hdd/scratch/dataset/mini_imagenet_split/labels/train_label.npy"
flo_label = f"/export/hdd/scratch/dataset/102flowers/labels/train_label.npy"
cifar_label = f"/export/hdd/scratch/dataset/cifar_10_images/labels/cifar_10_labels.npy"

mini_weight = f"/export/hdd/scratch/dataset/INR_weights/mini_imagenet/"
flo_weight = f"/export/hdd/scratch/dataset/INR_weights/102flowers/trail_10x32_30fq_5000ep_255im/"
cifar_weight = f"/export/hdd/scratch/dataset/INR_weights/cifar_10/trail_5x13_30fq_5000ep_12500im/"

for i in range(1, 9):
    print("flower, num_worker=", i)
    # flower
    with open("log/baseline_flo_%d.log" % i, "w") as out:
        sp.run(shlex.split("python 102flowers_baseline.py -nw %d" % (i)), stdout=out)

    # # cifar
    print("cifar, num_worker=", i)
    with open("log/baseline_cifar_%d.log" % i, "w") as out:
        sp.run(shlex.split("python cifar10_baseline.py -nw %d" % (i)), stdout=out)
    # sp.Popen(shlex.split(
    #     "python cifar10_baseline.py -nw %d > log/baseline_cifar10_%d.log " % (i, i)))
    # # mini imagenet
    print("mini, num_worker=", i)
    with open("log/baseline_mini_%d.log" % i, "w") as out:
        sp.run(shlex.split("python datamove.py -nw %d" % (i)), stdout=out)
    # sp.Popen(shlex.split(
    #     "python datamove.py -nw %d > log/baseline_mini_%d.log " % (i, i)))

for i in range(1, 9):
    print("flower, num_thread=", i)
    # flower
    with open("log/dali_flo_%d.log" % i, "w") as out:
        sp.run(shlex.split("python dali_flower.py -nt %d" % (i)), stdout=out)

    # # cifar
    print("cifar, num_worker=", i)
    with open("log/dali_cifar_%d.log" % i, "w") as out:
        sp.run(shlex.split("python dali_cifar.py -nt %d" % (i)), stdout=out)
    # sp.Popen(shlex.split(
    #     "python cifar10_baseline.py -nw %d > log/baseline_cifar10_%d.log " % (i, i)))
    # # mini imagenet
    print("mini, num_worker=", i)
    with open("log/dali_mini_%d.log" % i, "w") as out:
        sp.run(shlex.split("python dali_mini.py -nt %d" % (i)), stdout=out)
    # sp.Popen(shlex.split(
    #     "python datamove.py -nw %d > log/baseline_mini_%d.log " % (i, i)))
print("finished")
