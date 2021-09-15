import FL_utils
import IID_fedAvg as IID
import non_IID_fedAvg as non_IID
import medical_preprocessing_2d
import SGD
import mailSender
import numpy
import tensorflow

def main():

    try:
        results = medical_preprocessing_2d.generateAndStore()
        string = ("evaluation of the dataset generated: " + str(results))

        print(string)
        mailSender.sendResults(False, string)

    except Exception as e:
        mailSender.sendResults(True, None)


main()
