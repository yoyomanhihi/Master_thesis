import os, shutil
import pydicom as pdc


def find_folder(path):
    files = os.listdir(path)
    for file in files:
        newpath = os.path.join(path, file)
        if len(os.listdir(newpath)) > 5:
            return newpath

def rename1():
    path = "NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics"
    files = os.listdir(path)
    files.sort()
    for f in files:
        newpath = path + "/" + f
        if os.path.isdir(newpath):
            newfiles = os.listdir(newpath)
            for f2 in newfiles:
                newpath2 = newpath + "/" + f2
                newfiles2 = os.listdir(newpath2)
                for f3 in newfiles2:
                    newpath3 = newpath2 + "/" + f3
                    newfiles3 = os.listdir(newpath3)
                    if len(newfiles3) > 5:
                        for f4 in newfiles3:
                            newpath4 = newpath3 + "/" + f4
                            read = pdc.dcmread(newpath4)
                            img_ID = read.SOPInstanceUID
                            new_name = str(img_ID) + ".dcm"
                            old_file = os.path.join(newpath3, f4)
                            new_file = os.path.join(newpath3, new_name)
                            os.rename(old_file, new_file)


def rename2():
    path = "NSCLC-Radiomics-Interobserver1/NSCLC-Radiomics-Interobserver1"
    files = os.listdir(path)
    files.sort()
    for f in files:
        newpath = path + "/" + f
        if os.path.isdir(newpath):
            newfiles = os.listdir(newpath)
            for f2 in newfiles:
                newpath2 = newpath + "/" + f2
                newfiles2 = os.listdir(newpath2)
                for f3 in newfiles2:
                    newpath3 = newpath2 + "/" + f3
                    newfiles3 = os.listdir(newpath3)
                    if len(newfiles3) > 5:
                        for f4 in newfiles3:
                            newpath4 = newpath3 + "/" + f4
                            read = pdc.dcmread(newpath4)
                            img_ID = read.SOPInstanceUID
                            new_name = str(img_ID) + ".dcm"
                            old_file = os.path.join(newpath3, f4)
                            new_file = os.path.join(newpath3, new_name)
                            os.rename(old_file, new_file)

def move_files1():
    path = "NSCLC-Radiomics/manifest-1603198545583/NSCLC-Radiomics"
    files = os.listdir(path)
    files.sort()
    for f in files:
        newpath = path + "/" + f
        if os.path.isdir(newpath):
            newfiles = os.listdir(newpath)
            for f2 in newfiles:
                newpath2 = newpath + "/" + f2
                newfiles2 = os.listdir(newpath2)
                main_folder = find_folder(newpath2)
                for f3 in newfiles2:
                    newpath3 = newpath2 + "/" + f3
                    newfiles3 = os.listdir(newpath3)
                    if len(newfiles3) < 5 and "Segmentation" not in newpath3:
                        for f4 in newfiles3:
                            newpath4 = os.path.join(newpath3, f4)
                            shutil.move(newpath4, main_folder)


def move_files2():
    path = "NSCLC-Radiomics-Interobserver1/NSCLC-Radiomics-Interobserver1"
    files = os.listdir(path)
    files.sort()
    for f in files:
        newpath = path + "/" + f
        if os.path.isdir(newpath):
            newfiles = os.listdir(newpath)
            for f2 in newfiles:
                newpath2 = newpath + "/" + f2
                newfiles2 = os.listdir(newpath2)
                main_folder = find_folder(newpath2)
                for f3 in newfiles2:
                    newpath3 = newpath2 + "/" + f3
                    newfiles3 = os.listdir(newpath3)
                    if len(newfiles3) < 5 and "Segmentation" not in newpath3:
                        for f4 in newfiles3:
                            newpath4 = os.path.join(newpath3, f4)
                            shutil.move(newpath4, main_folder)

move_files1()

