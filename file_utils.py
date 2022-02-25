import os

def read_measures(filepath):
    """ Read all lines of the file and store it in a list of int
        :param filepath: the file containing the measures
        :return: list of int containing the measures of the file
    """
    lines = []
    print(filepath)
    if(os.path.exists(filepath)):
        with open(filepath) as f:
            lines = f.readlines()

        for i in range(len(lines)):
            lines[i] = float(lines[i])

    return lines


def write_measures(filepath, data):
    for i in range(len(data)):
        f = open(filepath, "a")
        f.write(str(data[i]) + "\n")
        f.close()
