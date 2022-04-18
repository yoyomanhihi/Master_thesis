import os
import unet_running
import mailSender
import sys


name = sys.argv[1]
print('name: ' + str(name))

def main():

    try:
        # results = med_prep_2d.generateAndStore('2d_dataset_1.pickle', nbclients=300)
        results = unet_running.build_and_save_fedavg_2(datasetpath='datasets/dataset_heart_fedAvg', nbclients=3, name=name)
        # string = ("accuracy with lr: " + str(results))

        # print(string)
        mailSender.sendResults(False, None)

    except Exception as e:
        mailSender.sendResults(True, None)

# main()
unet_running.build_and_save_fedavg_2(datasetpath='datasets/dataset_lung_fedAvg50', preloaded="fedeq_1.h5", nbclients=3, name=name)
# unet_running.build_and_save(datasetpath='datasets/dataset_lung_fedAvg0/0', preloaded = None, epochs=100, name=name)