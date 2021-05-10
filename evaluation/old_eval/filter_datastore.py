# call this script with `python -m playground.create_datastore`
import os
import pickle
import numpy as np

DATASTORE_FILE = os.path.join(os.path.dirname(__file__), "../rebalanced_datastore_icon_title_2.pkl")
datastore = pickle.load(open(DATASTORE_FILE, "rb"))

removeClasses = ["rape"]
filtered_datastore = {}
for key in datastore.keys():
    d = datastore[key]
    cn = d["row"]["class"]
    #print(cn)
    if(cn not in removeClasses):
        filtered_datastore[key] = d
        print("add", cn, key)

pickle.dump(filtered_datastore, open("rebalanced_datastore_icon_title_2_noRape.pkl", "wb"))
print(len(filtered_datastore.keys()))