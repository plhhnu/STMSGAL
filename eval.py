import sklearn.metrics as metrics
import csv
import datetime
import numpy as np


def Spatialeval(output_CSV,X,y,npcs,dav,cal,sil,sdbw,table,parameter):
    # dav = dav
    # cal = cal
    # sil = sil
    c = len(np.unique(np.array(y)))
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(output_CSV, mode='a+') as f:
        f_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow([str(npcs), str(c),str(table),
        str(dav), str(cal), str(sil),str(sdbw),str(now),str(parameter)])

