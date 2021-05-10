import pickle
import os
import numpy as np
import pandas as pd

if os.uname().nodename == 'MBP-von-Tilman':
    COMPOELEM_ROOT = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/compoelem"
elif os.uname().nodename == 'lme117':
    COMPOELEM_ROOT = "/home/zi14teho/compositional_elements"
else:
    COMPOELEM_ROOT = os.getenv('COMPOELEM_ROOT')
#EVAL_RESULTS_FILE = COMPOELEM_ROOT+"/final_evaluation/evaluation_log_grid_tune.pkl"
EVAL_RESULTS_FILE = COMPOELEM_ROOT+"/final_evaluation/evaluation_log.pkl"

evaluation_log = []
[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth150_fb", "rb"))]
[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth150_noFb", "rb"))]
[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth200_fb", "rb"))]
[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth200_noFb", "rb"))]
[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth250_fb", "rb"))]
[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth250_noFb", "rb"))]
[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth300_fb", "rb"))]
[evaluation_log.append(le) for le in pickle.load(open(".tmpEvalLog_fth300_noFb", "rb"))]

# cone_opening_angle
# cone_scale_factor
# cone_base_scale_factor
# filter_threshold
x = np.array([le["filter_threshold"] for le in evaluation_log])
y = np.array([le["cone_scale_factor"] for le in evaluation_log])
z = np.array([le["eval_dataframe"].loc["total (mean)", "p@1"] for le in evaluation_log])
c = np.array([le["cone_base_scale_factor"] for le in evaluation_log])

with_fallback = np.array([le["with_fallback"] for le in evaluation_log])
cone_opening_angle = np.array([le["cone_opening_angle"] for le in evaluation_log])


# for cone_base_scale_factor in [0, 1, 2, 2.5]:
# for cone_scale_factor in [5, 10, 15]:
# for cone_opening_angle in [70, 80, 90]:
# for correction_angle in [40, 50]:

print("\nmeans cone_base_scale_factor:")
print("  0",np.mean(z[c == 0]))
print("  1",np.mean(z[c == 1]))
print("  2",np.mean(z[c == 2]))
print("2.5",np.mean(z[c == 2.5]))
print("\nmeans cone_scale_factor:")
print(" 5",np.mean(z[y == 5]))
print("10",np.mean(z[y == 10]))
print("15",np.mean(z[y == 15]))
print("\nmeans cone_opening_angle:")
print("70",np.mean(z[cone_opening_angle == 70]))
print("80",np.mean(z[cone_opening_angle == 80]))
print("90",np.mean(z[cone_opening_angle == 90]))
print("\nmeans filter_threshold:")
print("150",np.mean(z[x == 150]))
print("200",np.mean(z[x == 200]))
print("250",np.mean(z[x == 250]))
print("300",np.mean(z[x == 300]))
print("\nmeans fallback:")
print("wFb",np.mean(z[with_fallback == True]))
print("nFb",np.mean(z[with_fallback == False]))

print("\nmaxs cone_base_scale_factor:")
print("  0",np.max(z[c == 0]))
print("  1",np.max(z[c == 1]))
print("  2",np.max(z[c == 2]))
print("2.5",np.max(z[c == 2.5]))
print("\nmaxs cone_scale_factor:")
print(" 5",np.max(z[y == 5]))
print("10",np.max(z[y == 10]))
print("15",np.max(z[y == 15]))
print("\nmaxs cone_opening_angle:")
print("70",np.max(z[cone_opening_angle == 70]))
print("80",np.max(z[cone_opening_angle == 80]))
print("90",np.max(z[cone_opening_angle == 90]))
print("\nmaxs filter_threshold:")
print("150",np.max(z[x == 150]))
print("200",np.max(z[x == 200]))
print("250",np.max(z[x == 250]))
print("300",np.max(z[x == 300]))
print("\nmaxs fallback:")
print("wFb",np.max(z[with_fallback == True]))
print("nFb",np.max(z[with_fallback == False]))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
fig.colorbar(img)
plt.show()