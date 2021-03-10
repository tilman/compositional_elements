# call this script with `python -m playground.create_datastore`
import os
import pickle
import numpy as np

DATASTORE_FILE = os.path.join(os.path.dirname(__file__), "./datastore_icon_title_2.pkl")
try:
    datastore = pickle.load(open(DATASTORE_FILE, "rb"))
except FileNotFoundError as e:
    datastore = {}

SIZE = 100
classes = {
    #classname: keys[]
    "annunciation":[],
    "nativity":[],
    "adoration":[],
    "baptism":[],
    #"rape":[],
    "virgin and child":[],
}

for key in datastore.keys():
    d = datastore[key]
    cn = d["row"]["class"]
    if(cn in classes):
        classes[cn].append(key)

rebalanced_datastore = {}

for c in classes.keys():
    keys = np.array(classes[c])
    random_keys = keys[np.random.choice(len(keys), size=SIZE, replace=False)]
    for rk in random_keys:
        rebalanced_datastore[rk] = datastore[rk]
#v3 get rid of rape class
pickle.dump(rebalanced_datastore, open("./rebalanced_datastore_icon_title_3.pkl", "wb"))
