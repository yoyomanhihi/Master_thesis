import FL_utils
import IID_fedAvg as IID
import non_IID_fedAvg as non_IID
import medical_preprocessing_2d as med_prep_2d
import medical_SGD as med_SGD
import mailSender
import numpy
import tensorflow

def main():

    try:
        # results = med_prep_2d.generateAndStore('2d_dataset_1.pickle', nbclients=300)
        results = med_SGD.make_all()
        string = ("accuracy full tumor with exact arrays as input: " + str(results))

        print(string)
        mailSender.sendResults(False, string)

    except Exception as e:
        mailSender.sendResults(True, None)


main()

# med_SGD.make_all()
# med_prep_2d.generateAndStore('2d_dataset_1.pickle', nbclients=300)