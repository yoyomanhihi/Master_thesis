import unet_running
import mailSender
import sys


name = sys.argv[1]
print('name: ' + str(name))

def main():

    try:
        # results = med_prep_2d.generateAndStore('2d_dataset_1.pickle', nbclients=300)
        results = unet_running.build_and_save(datasetpath='datasets/dataset_heart_fedAvg/0', epochs=100, name=name)
        # string = ("accuracy with lr: " + str(results))

        # print(string)
        mailSender.sendResults(False, None)

    except Exception as e:
        mailSender.sendResults(True, None)

# main()
# unet_running.build_and_save_fedavg(datasetpath='datasets/dataset_heart_fedAvg', nbclients=3, name=name)
unet_running.build_and_save(datasetpath='datasets/dataset_heart_fedAvg/1', epochs=100, name=name, preloaded = 'models/ds1_heart_39epochs.h5')