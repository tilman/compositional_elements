import pickle
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
import shutil
import pandas as pd
from plotnine import ggplot, aes, geom_line, ylim, theme, ggtitle
from plotnine.scales import limits, scale_color_manual, scale_linetype_manual, scale_y_continuous, scale_x_continuous
from plotnine.ggplot import ggsave, save_as_pdf_pages
from plotnine.themes.elements import element_text

from compoelem.detect import converter
from compoelem.config import config
from compoelem.generate import global_action, pose_abstraction
from compoelem.visualize import visualize

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
DATASET_ROOT = "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/datasets/dataset_cleaned_extended_balanced"

datastore = pickle.load(open(DATASTORE_FILE, "rb"))
datastore_name = DATASTORE_NAME

evaluation_log = [pickle.load(open(EVAL_RESULTS_FILE_DIR+"/"+logfile, "rb")) for logfile in os.listdir( EVAL_RESULTS_FILE_DIR )]
# print(len(evaluation_log))

display_metrics = ["p@1","p@2","p@5","p@10"]
a = pd.DataFrame([
    [
        le,
        le['experiment_name'],
        le["compare_other"] if "compare_other" in le else None, 
        le['filename'][12:-4],
        le['datetime'].strftime("%d.%m.%y"),
        *le["eval_dataframe"].loc["total (mean)", display_metrics],
    ] for le in evaluation_log 
      if le['datetime'].strftime("%d.%m.%y") == "20.04.21"
    ], columns=["log_entry", "experiment_name", "compare_other", "name", "datetime", *display_metrics]).sort_values("datetime")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
a = a[a['experiment_name'].str.contains("plots")]

def plot_retrievals(query_label, query_key, retrieval_labels, retrieval_keys, experiment_name):
    fig, axis = plt.subplots(1, len(retrieval_keys)+1)

    className = query_key.split('_')[0]
    imgName = query_key[len(className)+1:len(query_key)]
    filename = DATASET_ROOT+'/'+className+'/'+imgName
    query_img = cv2.imread(filename)

    query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    ax = axis[0] # type: ignore
    ax.imshow(query_img_rgb)
    ax.axis('off')
    ax.set_title('{}\nquery\n{}'.format(experiment_name, query_label))

    for res_idx, res_key, res_label in zip(range(0, len(retrieval_keys)), retrieval_keys, retrieval_labels): #type: ignore
        #res_img = converter.resize(res_img)
        #res_img = visualize.pose_lines(res_data["pose_lines"], res_img) # type: ignore
        #res_img = visualize.global_action_lines(res_data["global_action_lines"], res_img) # type: ignore
        className = res_key.split('_')[0]
        imgName = res_key[len(className)+1:len(res_key)]
        filename = DATASET_ROOT+'/'+className+'/'+imgName
        res_img = cv2.imread(filename)
        res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        ax = axis[res_idx+1]
        ax.imshow(res_img_rgb)
        ax.axis('off')
        ax.set_title(res_label)
    
    plt.tight_layout()
    #plt.subplots_adjust(top=0.85) # Make space for title
    plt.show()


COUNT_QUALITY_LEN = 10
COUNT_DISPLAY_LEN = 5

def get_top_result(data):
    experiment_name = data["experiment_name"]
    #print("\nexperiment_name:",experiment_name)
    all_retrieval_res = data["log_entry"]["all_retrieval_res"]
    try:
        pose_config = (
            data["log_entry"]["correction_angle"],
            data["log_entry"]["cone_opening_angle"],
            data["log_entry"]["cone_scale_factor"],
            data["log_entry"]["cone_base_scale_factor"],
            data["log_entry"]["filter_threshold"],
            False, #data["log_entry"]["bisection_fallback"],
            True, #data["log_entry"]["poseline_fallback"],
        )
    except:
        # default baseline config for deep or traditional only methods
        pose_config = (
            20,
            80,
            10,
            0,
            150,
            False,
            True,
        )

    # TODO: extend this to best retrieval per class
    best_retrieval_sum = 0
    worst_retrieval_sum = 1000
    best_retrieval_pair = ()
    worst_retrieval_pair = ()
    for retrieval_res in all_retrieval_res:
        query_key, query_label, retrieval_keys, retrieval_labels = retrieval_res
        if query_label == "annunciation":
            matched_retrievals = sum([query_label == r for r in retrieval_labels[0:COUNT_QUALITY_LEN]])
            if matched_retrievals > best_retrieval_sum:
                res_poses = datastore[query_key]["compoelem"]["poses"]
                if len(res_poses) > 0:
                    best_retrieval_sum = matched_retrievals
                    #print("best_retrieval_sum", best_retrieval_sum)
                    best_retrieval_pair = (query_label, query_key, retrieval_labels[0:COUNT_DISPLAY_LEN], retrieval_keys[0:COUNT_DISPLAY_LEN])
                else:
                    #print("res_poses has len 0", res_poses)
                    pass
            if matched_retrievals < worst_retrieval_sum:
                worst_retrieval_sum = matched_retrievals
                #print("worst_retrieval_sum", worst_retrieval_sum)
                worst_retrieval_pair = (query_label, query_key, retrieval_labels[0:COUNT_DISPLAY_LEN], retrieval_keys[0:COUNT_DISPLAY_LEN])
    #print(best_retrieval_pair, worst_retrieval_pair)
    #plot_retrievals(*best_retrieval_pair, experiment_name)
    return best_retrieval_pair, worst_retrieval_pair, experiment_name, pose_config


LATEX_ROOT_DIR = '/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/thesis_latex/'
LATEX_PLOT_OUTPUT_DIR = 'images/evaluation/retrieval_plots/'
LATEX_SUBFIGURE_IDX = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y"]

def create_latex_code(query_label, query_key, retrieval_labels, retrieval_keys, experiment_name, pose_config):
    save_experiment_name = experiment_name.replace(':','_').replace(' ','_').replace('+','_').replace('&','_')
    PLOT_DIR = LATEX_PLOT_OUTPUT_DIR+save_experiment_name+'/' #main plot dir
    ROOT_DIR = LATEX_ROOT_DIR+PLOT_DIR
    try:
        os.mkdir(ROOT_DIR)
    except FileExistsError as e:
        pass

    correction_angle, cone_opening_angle, cone_scale_factor, cone_base_scale_factor, filter_threshold, bisection_fallback, poseline_fallback = pose_config
    config["bisection"]["correction_angle"] = correction_angle
    config["bisection"]["cone_opening_angle"] = cone_opening_angle
    config["bisection"]["cone_scale_factor"] = cone_scale_factor
    config["bisection"]["cone_base_scale_factor"] = cone_base_scale_factor

    queryImgName = query_key[len(query_label)+1:len(query_key)]
    queryFilename = query_label+'/'+queryImgName
    
    queryImg = cv2.imread(DATASET_ROOT+'/'+queryFilename)
    queryImg = converter.resize(queryImg)
    query_poses = datastore[query_key]["compoelem"]["poses"]
    query_global_action_lines = global_action.get_global_action_lines(query_poses, False)
    query_poselines = pose_abstraction.get_pose_lines(query_poses, True)
    queryImg = visualize.pose_lines(query_poselines, queryImg)
    queryImg = visualize.global_action_lines(query_global_action_lines, queryImg)
    cv2.imwrite(ROOT_DIR+queryImgName, queryImg)



    latex_graphic_size_query = "width=0.17\\linewidth"
    latex_graphic_size = "width=0.155\\linewidth"
    latex_caption = ""
    latex = "\\begin{figure}\n\\centering\n\\subfloat[][]{\\includegraphics["+latex_graphic_size_query+"]{"+PLOT_DIR+queryImgName+"}}\\qquad"
    for res_idx, res_key, res_label in zip(range(0, len(retrieval_keys)), retrieval_keys, retrieval_labels):
        # resClassName = res_key.split('_')[0]
        resImgName = res_key[len(res_label)+1:len(res_key)]
        resFilename = res_label+'/'+resImgName
        resImg = cv2.imread(DATASET_ROOT+'/'+resFilename)
        resImg = converter.resize(resImg)
        res_poses = datastore[res_key]["compoelem"]["poses"]
        res_global_action_lines = global_action.get_global_action_lines(res_poses, False)
        res_poselines = pose_abstraction.get_pose_lines(res_poses, True)
        resImg = visualize.pose_lines(res_poselines, resImg)
        resImg = visualize.global_action_lines(res_global_action_lines, resImg)
        cv2.imwrite(ROOT_DIR+resImgName, resImg)

        # TODO: add title to image
        latex += "\n\\subfloat[][]{\\includegraphics["+latex_graphic_size+"]{"+PLOT_DIR+resImgName+"}}"
        latex_caption += "\n"+LATEX_SUBFIGURE_IDX[res_idx]+") "+res_label+": "+resImgName
    
    latex += "\n\\caption{\nExperiment: "+experiment_name.replace('_','\\_').replace('&','\\&')+latex_caption.replace('_','\\_')+"\n}\n\\label{fig:eval:"+save_experiment_name+"}\n\\end{figure}\n\n"
    print(latex)


print("\n\n\n\n\n\n")
print("\n\n\n\n\n\n")
for i in range(0, len(a)):
    best_retrieval_pair, worst_retrieval_pair, experiment_name, pose_config = get_top_result(a.iloc[i])
    create_latex_code(*best_retrieval_pair, experiment_name, pose_config)
print("\n\n\n\n\n\n")

