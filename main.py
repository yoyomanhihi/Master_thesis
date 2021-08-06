import local_FL_utils as FL_utils
import IID_fedAvg as IID
import non_IID_fedAvg as non_IID
import SGD
import mailSender

def main():
    try:
        results = SGD.cross_val_SGD()
        print(results)

        mailSender.sendResults(False, results)

    except Exception as e:
        mailSender.sendResults(True, None)

print(SGD.simpleSGDTest())