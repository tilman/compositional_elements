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
    #print(a[["experiment_name","name","datetime", *display_metrics]])
    #a = a[a['name'].str.contains("70421134822_normGlac_hr_nmd_desc_ca20_co80_cs10_cbs0_th150_fbPlTrue_fbBisFalse_fbGaTrue_otherNone_aw0.5")]
    #a = a[a['name'].str.contains("70421134822_normGlac_hr_nmd_desc_ca20_co80_cs10_cbs0_th150_fbPlTrue_fbBisFalse_fbGaTrue_otherNone_aw0.5")]
    #print(a[["experiment_name","name","datetime", *display_metrics]])
    # curves = a.sort_values("datetime").iloc[-1]["log_entry"]["precision_curves"]
    # return curves


all_means = {}

def get_precsion_plot(data, legend):
    global all_means
    curves = data["log_entry"]["precision_curves"]
    means = {}
    mean_all = []
    for className in curves:
        precision_at_rank = np.array(curves[className])
        means[className] = np.mean(precision_at_rank, axis=0)
        mean_all.append(means[className])
    means["total"] = np.mean(mean_all, axis=0)
    print(means["total"][0:10])

    mpd = pd.DataFrame([{"class": c, "precision":p, "rank":r+1} for c in means.keys() for r,p in enumerate(means[c]) ])
    title = data["experiment_name"]
    title = title[7:len(title)]

    all_means[title]=means["total"]

    print("title",title,legend)
    if legend:
        base = ggplot(mpd, aes(x = "rank", y = "precision")) + theme(figure_size=(4,2.8))
    else:
        base = ggplot(mpd, aes(x = "rank", y = "precision")) + theme(legend_position = "none", figure_size=(3.5,2.8))
    return (
        base
        + ggtitle(title)
        #+ geom_line(aes( color = "class", linetype="class", w=1.4))
        + geom_line(aes( color = "class", linetype="class"))
        #+ scale_color_manual({"total":"black","adoration":"red","annunciation":"red","baptism":"red","virgin and child":"red","nativity":"red"})
        + scale_linetype_manual(values=["dashed","dashed","dashed","dashed","solid","dashed"])
        #+ ylim(0.1, 0.9)
        + scale_y_continuous(limits=(0.15,0.85), breaks=[0.2,0.3,0.4,0.5,0.6,0.7,0.8])
        + scale_x_continuous(limits=(1,50), breaks=[1,10,20,30,40,50])
        + theme(plot_title = element_text(size=11) )
    )

# first generate all individual plots, with and without legend
precision_plots = [
    get_precsion_plot(a.iloc[i], legend)
    for i in range(0,len(a))
    for legend in [True, False]
]
save_as_pdf_pages(precision_plots, filename="/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/thesis_latex/images/ggplot_output/eval/precision_plots.pdf")


def get_precsion_pair_plot(data1, data2, legend):
    curves1 = data1["log_entry"]["precision_curves"]
    curves2 = data2["log_entry"]["precision_curves"]

    means1 = {}
    mean_all1 = []
    for className in curves1:
        precision_at_rank = np.array(curves1[className])
        means1[className] = np.mean(precision_at_rank, axis=0)
        mean_all1.append(means1[className])
    means1["total"] = np.mean(mean_all1, axis=0)
    means2 = {}
    mean_all2 = []
    for className in curves2:
        precision_at_rank = np.array(curves2[className])
        means2[className] = np.mean(precision_at_rank, axis=0)
        mean_all2.append(means2[className])
    means2["total"] = np.mean(mean_all2, axis=0)

    method1 = data1["experiment_name"]
    method1 = method1[7:len(method1)]
    method2 = data2["experiment_name"]
    method2 = method2[7:len(method2)]
    methods = [method1, method2]

    print("means1&2", means1["total"][0:2], means2["total"][0:2])

    # split into 2 times 3 classes for better visibility
    mpd1 = pd.DataFrame([{"method": methods[mi], "class": c, "precision":p, "rank":r+1} for mi, means in enumerate([means1, means2]) for c in list(means.keys())[0:3] for r,p in enumerate(means[c]) ])
    # split into 2 times 3 classes for better visibility
    mpd2 = pd.DataFrame([{"method": methods[mi], "class": c, "precision":p, "rank":r+1} for mi, means in enumerate([means1, means2]) for c in list(means.keys())[3:6] for r,p in enumerate(means[c]) ])

    #title = ""
    print("methods",methods,legend)
    if legend:
        base1 = ggplot(mpd1, aes(x = "rank", y = "precision")) + theme(figure_size=(4,2.8))
        base2 = ggplot(mpd2, aes(x = "rank", y = "precision")) + theme(figure_size=(4,2.8))
    else:
        base1 = ggplot(mpd1, aes(x = "rank", y = "precision")) + theme(legend_position = "none", figure_size=(3.5,2.8))
        base2 = ggplot(mpd2, aes(x = "rank", y = "precision")) + theme(legend_position = "none", figure_size=(3.5,2.8))
    return [
        (
            base1
            #+ ggtitle(title)
            #+ geom_line(aes( color = "class", linetype="class", w=1.4))
            + geom_line(aes( color="class", linetype="method" ))
            #+ scale_color_manual({"total":"black","adoration":"red","annunciation":"red","baptism":"red","virgin and child":"red","nativity":"red"})
            #+ scale_linetype_manual(values=["dashed","dashed","dashed","dashed","solid","dashed"])
            #+ ylim(0.1, 0.9)
            + scale_y_continuous(limits=(0.15,0.85), breaks=[0.2,0.3,0.4,0.5,0.6,0.7,0.8])
            + scale_x_continuous(limits=(1,50), breaks=[1,10,20,30,40,50])
            #+ theme(plot_title = element_text(size=11) )
        ), (
            base2
            #+ ggtitle(title)
            #+ geom_line(aes( color = "class", linetype="class", w=1.4))
            + geom_line(aes( color="class", linetype="method" ))
            #+ scale_color_manual({"total":"black","adoration":"red","annunciation":"red","baptism":"red","virgin and child":"red","nativity":"red"})
            #+ scale_linetype_manual(values=["dashed","dashed","dashed","dashed","solid","dashed"])
            #+ ylim(0.1, 0.9)
            + scale_y_continuous(limits=(0.15,0.85), breaks=[0.2,0.3,0.4,0.5,0.6,0.7,0.8])
            + scale_x_continuous(limits=(1,50), breaks=[1,10,20,30,40,50])
            #+ theme(plot_title = element_text(size=11) )
        )
    ]
precision_pair_plots = [
    get_precsion_pair_plot(
        a[a['experiment_name'].str.contains(si)].iloc[0],
        a[a['experiment_name'].str.contains(sj)].iloc[0],
        legend
    )
    for si,sj in [
        ("ICC\+ U AR","ICC\+ T AR"),
        
        ("ICC\+ T AR \& VGG19 combi1_asc","ICC\+ T AR"),
        ("ICC\+ T AR \& ResNet50 combi1_asc","ICC\+ T AR"),

        ("ICC\+ T AR \& ResNet50 combi1_asc","ICC\+ T AR \& VGG19 combi1_asc"),
    ]
    for legend in [True, False]
]
save_as_pdf_pages(np.array(precision_pair_plots).flatten(), filename="/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/thesis_latex/images/ggplot_output/eval/precision_pair_plots.pdf")

# then generate combined plot
def get_combined_plot(legend):
    global all_means

    mpd = pd.DataFrame([{"method": c, "precision":p, "rank":r+1} for c in all_means.keys() if "combi2_asc" not in c for r,p in enumerate(all_means[c]) ])
    title = "all methods"

    print("title",title,legend)
    if legend:
        base = ggplot(mpd, aes(x = "rank", y = "precision")) + theme(figure_size=(4,2.8))
    else:
        base = ggplot(mpd, aes(x = "rank", y = "precision")) + theme(legend_position = "none", figure_size=(3.5,2.8))
    return (
        base
        + ggtitle(title)
        + geom_line(aes( color = "method"))
        + scale_y_continuous(limits=(0.25,0.65), breaks=[0.2,0.3,0.4,0.5,0.6])
        + scale_x_continuous(limits=(1,50), breaks=[1,10,20,30,40,50])
        + theme(plot_title = element_text(size=11) )
    )

combined_plots = [
    get_combined_plot(legend)
    for legend in [True, False]
]
save_as_pdf_pages(combined_plots, filename="/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/thesis_latex/images/ggplot_output/eval/combined_plots.pdf")