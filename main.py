import FL_utils
import IID_fedAvg as IID
import non_IID_fedAvg as non_IID
import SGD
import mailSender
import numpy
import tensorflow

def main():
    '''
    try:
        results = non_IID.cross_val_fedAvg()
        string = ("non IID FedAvg cross val: " + str(results))

        print(string)
        mailSender.sendResults(False, string)

    except Exception as e:
        mailSender.sendResults(True, None)
    '''
    IID.simpleFedAvgTest()
main()
