import sklearn.metrics as metrics
import csv
import datetime
import numpy as np

def title(output_CSV):
    with open(output_CSV, mode='a+') as f:
        f_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow(["dim","n_clusters","knn","cluster_table","beta",'ann',
        "dav", "cal", "sil","time"])
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
def Spatialmean(output_meanCSV,davmean,calmean,silmean,npcs,pred_label,n):
    davmean = np.round(davmean, 5)
    calmean = np.round(calmean, 5)
    silmean = np.round(silmean, 5)
    c = len(np.unique(np.array(pred_label)))
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(output_meanCSV, mode='a+') as f:
        f_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        f_writer.writerow([str(npcs), str(c),str(n), 
        str(davmean), str(calmean), str(silmean),str(now)])
