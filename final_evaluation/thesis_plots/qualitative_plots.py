import pickle
import os
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_line, ylim, theme, ggtitle
from plotnine.scales import limits, scale_color_manual, scale_linetype_manual, scale_y_continuous, scale_x_continuous
from plotnine.ggplot import ggsave, save_as_pdf_pages
from plotnine.themes.elements import element_text

# def get_curves():
osuname = os.uname().nodename
print("osuname", osuname)
if osuname == 'MBP-von-Tilman' or osuname == 'MacBook-Pro-von-Tilman.local':
    COMPOELEM_ROOT = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/compoelem"
elif osuname == 'lme117':
    COMPOELEM_ROOT = "/home/zi14teho/compositional_elements"
else:
    COMPOELEM_ROOT = os.getenv('COMPOELEM_ROOT')
DATASTORE_NAME = "combined_datastore_ceb_dataset"
DATASTORE_FILE = COMPOELEM_ROOT+"/final_evaluation/"+DATASTORE_NAME+".pkl"
EVAL_RESULTS_FILE_DIR = COMPOELEM_ROOT+"/final_evaluation/final2pkl/"

evaluation_log = [pickle.load(open(EVAL_RESULTS_FILE_DIR+"/"+logfile, "rb")) for logfile in os.listdir( EVAL_RESULTS_FILE_DIR )]
print(len(evaluation_log))

display_metrics = ["p@1","p@2","p@5","p@10"]
a = pd.DataFrame([
    [
        le,
        le['experiment_name'],
        le["compare_other"] if "compare_other" in le else None, 
        le['filename'][12:-4],
        le['datetime'].strftime("%d.%m.%y %H:%M"),
        *le["eval_dataframe"].loc["total (mean)", display_metrics],
    ] for le in evaluation_log], columns=["log_entry", "experiment_name", "compare_other", "name", "datetime", *display_metrics]).sort_values("datetime")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
a = a[a['experiment_name'].str.contains("plots")]

def get_top_result(data):
    curves = data["log_entry"]["precision_curves"]
    curves.shape
    for key in curves.keys():
        curve = np.array(curves["adoration"])
        # TODO: filter curve for max..
        # TODO: extend precision curve with retrieval index, or use retrieval labels..

print(get_top_result(a.iloc[0]))