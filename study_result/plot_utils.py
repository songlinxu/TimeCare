import os, sys, time
import pandas as pd  
import numpy as np 
from numpy.random import seed
from sklearn import preprocessing
from scipy import stats
from matplotlib.cbook import boxplot_stats
import seaborn as sns
import matplotlib.pyplot as plt
import json

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params, font_scale = 1.5)

def get_user(dataset):
    return list(set(dataset['user_id']))

def visual_result(dataset_path,datatype,plottype):
    fig, ax = plt.subplots(figsize=(7,5))
    dataset = pd.read_csv(dataset_path)

    color_dict = {'accuracy_abs': 'Greens','accuracy_rela': 'Greens','resptime_abs': 'Oranges','resptime_rela': 'Oranges','attention_abs': 'Blues', 'attention_rela': 'Blues', 'anxiety_abs': 'Reds', 'anxiety_rela': 'Reds'}

    assert plottype in ['box','hist']
    if plottype == 'box':
        sns.boxplot(data=dataset,x='group',y=datatype, flierprops = dict(markerfacecolor='w', marker='o'), width=0.75, palette=color_dict[datatype], boxprops=dict(alpha=.9))
    else:
        sns.histplot(data=dataset, x=datatype, hue='group', kde=True, stat = 'percent')

    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.show()


def all_bar_abs(dataset_path,group='random'):
    assert group in ['rl','random']
    
    dataset = pd.read_csv(dataset_path)
    dataset = dataset[dataset['group']==group]
    user_all = get_user(dataset)
    dataset_new = pd.DataFrame(columns = ['user_id','group','order','resptime_control','resptime_feedback'])
    for user_id in user_all:
        data_user = dataset[dataset['user_id']==user_id]
        order = data_user['order'].values[0]
        assert order in [0,1]
        if order == 0:
            dataset_new.loc[len(dataset_new.index)] = [user_id,data_user['group'].values[0],order,data_user['t1_resptime'].values[0],data_user['t2_resptime'].values[0]]
        else:
            dataset_new.loc[len(dataset_new.index)] = [user_id,data_user['group'].values[0],order,data_user['t2_resptime'].values[0],data_user['t1_resptime'].values[0]]
    dataset_new = dataset_new.sort_values("resptime_control", ascending=False)
    dataset_new.user_id = dataset_new.user_id.astype('string')
    
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(8, 5))
    blog_blue = 'g'
    blue_cmap = sns.light_palette(blog_blue, as_cmap=True)
    sns.set_color_codes("pastel")
    color_map = blue_cmap((dataset_new['resptime_control']-np.min(dataset_new['resptime_control']))/(np.max(dataset_new['resptime_control'])-np.min(dataset_new['resptime_control'])))
    ax=sns.barplot(x="resptime_control", y="user_id", data=dataset_new, label="resptime_control", color='b')


    blog_blue = 'b'
    blue_cmap = sns.light_palette(blog_blue, as_cmap=True)
    sns.set_color_codes("pastel")
    color_map2 = blue_cmap((dataset_new['resptime_control']-np.min(dataset_new['resptime_control']))/(np.max(dataset_new['resptime_control'])-np.min(dataset_new['resptime_control'])))


    sns.set_color_codes("muted")
    ax=sns.barplot(x="resptime_feedback", y="user_id", data=dataset_new, label="resptime_feedback", color='g', alpha=0.6)

    ax.set_box_aspect(0.5)

    if group == 'random':
        ax.legend(ncol=2, loc="lower left", frameon=True)
    else:
        ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(2, 10), ylabel="", xlabel="")
    plt.tick_params(labelleft = False)
    if group == 'random':
        ax.invert_xaxis()
    sns.despine(left=True, bottom=True)

    plt.show()



def all_bar_rela(dataset_path,group='rl'):
    dataset = pd.read_csv(dataset_path)
    dataset_group = dataset[dataset['group']==group]
    dataset_group = dataset_group.sort_values("resptime_rela", ascending=True)
    dataset_group.user_id = dataset_group.user_id.astype('string')

    blue_cmap = sns.light_palette('lightskyblue', as_cmap=True)  
    red_cmap = sns.light_palette('lightcoral', as_cmap=True) 
    
    negative_resptime = dataset_group[dataset_group['resptime_rela']<0]['resptime_rela']
    positive_resptime = dataset_group[dataset_group['resptime_rela']>=0]['resptime_rela']

    color_map_neg = blue_cmap((np.max(negative_resptime)-negative_resptime)/(np.max(negative_resptime)-np.min(negative_resptime)))
    color_map_pos = red_cmap((positive_resptime-np.min(positive_resptime))/(np.max(positive_resptime)-np.min(positive_resptime)))
    
    color_map_all = np.concatenate((color_map_neg,color_map_pos),axis=0)
 
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(14, 4))
    sns.barplot(x="user_id", y="resptime_rela", data=dataset_group,label="Relative Response Time Change", palette=color_map_all)

    ax.legend(ncol=2, loc="lower center", frameon=True)
    ax.set(ylabel="", xlabel="")
    ax.invert_yaxis()
    sns.despine(left=True, bottom=True)
    plt.tick_params(labelbottom = False)

    plt.show()




def visual_trend(dataset_path,datatype):
    fig, ax = plt.subplots(1,1)
    data_table_average = pd.read_csv(dataset_path)
    data_table_average.block = data_table_average.block + 1
    sns.lineplot(data=data_table_average, x='block', y=datatype, hue='group')
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.show()


