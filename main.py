import features
import tarfile
import os
import json

tar = tarfile.open("features.tar.gz", "r:gz")
fileeval=list()
filebalan=list()
fileunbalan=list()
for member in tar.getmembers():
    if "tfrecord" in member.name:
        p = os.path.dirname(os.path.abspath(member.name))
        x=member.name.split('/')
        if "eval" in member.name:
            fileeval.append(p+'/'+x[2])
        elif "unbal_train" in member.name:
            fileunbalan.append(p+'/'+x[2])
        elif "bal_train" in member.name:
            filebalan.append(p+'/'+x[2])
x_eval,Y_eval=features.estrattore_features_tensor(fileeval[0])
labeled_examples = list(zip(x_eval, Y_eval))
elem=json.dumps(labeled_examples)
f=open("prova.txt","wb")
f.write(elem)
