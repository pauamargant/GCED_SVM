import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

def plot_execution_time(df, parameter_name):
    primal_time = df[df["type"]=="primal"]["solve_time"]
    dual_lin_time = df[df["type"]=="dual_linear"]["solve_time"]
    dual_ker_time = df[df["type"]=="dual_gaussian"]["solve_time"]
    parameter_values = df[df["type"]=="primal"][parameter_name]

    fig = px.scatter(y=primal_time, x=parameter_values)
    fig.add_trace(go.Scatter(x=parameter_values, y=primal_time,
                    mode='markers',
                    name='Primal'))
                    
    fig.add_trace(go.Scatter(x=parameter_values, y=dual_lin_time,
                    mode='markers',
                    name='Dual (linear)'))
    fig.add_trace(go.Scatter(x=parameter_values, y=dual_ker_time,
                    mode='markers',
                    name='Dual (kernel)'))
    fig.update_layout(title=f"Solve time vs {parameter_name}")
    fig.update_xaxes(title_text=parameter_name)
    fig.update_yaxes(title_text="Solve time (s)")
    fig.update_traces(marker_size=10)
    fig.update_layout(legend_title_text="Nodes")
    fig.write_image("plots/time_vs_"+parameter_name+".pdf", width=600)
    fig.show()

def plot_accuracy(df, parameter_name):
    primal_accuracy = df[df["type"]=="primal"]["accuracy"]
    dual_lin_accuract = df[df["type"]=="dual_linear"]["accuracy"]
    dual_ker_accuracy = df[df["type"]=="dual_gaussian"]["accuracy"]
    parameter_values = df[df["type"]=="primal"][parameter_name]

    fig = px.scatter(y=primal_accuracy, x=parameter_values)
    fig.add_trace(go.Scatter(x=parameter_values, y=primal_accuracy,
                    mode='markers',
                    name='Primal'))
                    
    fig.add_trace(go.Scatter(x=parameter_values, y=dual_lin_accuract,
                    mode='markers',
                    name='Dual (linear)'))
    fig.add_trace(go.Scatter(x=parameter_values, y=dual_ker_accuracy,
                    mode='markers',
                    name='Dual (kernel)'))
    fig.update_layout(title=f"Accuracy vs {parameter_name}")
    fig.update_xaxes(title_text=parameter_name)
    fig.update_yaxes(title_text="Accuracy")
    fig.update_traces(marker_size=10)
    fig.update_layout(legend_title_text="Nodes")
    fig.write_image("plots/acc_vs_"+parameter_name+".pdf", width=600)
    fig.show()


def time_heatmap(df, save=False,color = "accuracy", multiple = False):
    # if color is solve_time use log scale
    
    if not multiple:
        fig = sns.heatmap(df.pivot("m", "nu", color), cbar=False,
                        annot=True, fmt=".2f", cmap="viridis")
        fig.set_xlabel("m")
        fig.set_ylabel("nu")
    else:
        df_primal = df[df['type']=='primal']
        df_dual_lin = df[df['type']=='dual_linear']
        df_dual_ker = df[df['type']=='dual_gaussian']
        # three heatmaps side by side
        fig, axs = plt.subplots(1, 4, figsize=(16, 5),gridspec_kw=dict(width_ratios=[5,5,5,1]))
        # import lognorm
        from matplotlib.colors import LogNorm
        sns.heatmap(df_primal.pivot("m", "nu", color),
                    annot=True, fmt=".2f",cbar=False,  cmap="viridis", ax=axs[0], norm = LogNorm())
        sns.heatmap(df_dual_lin.pivot("m", "nu", color),cbar=False, yticklabels=False, norm = LogNorm(),
                    annot=True, fmt=".2f", cmap="viridis", ax=axs[1])
        sns.heatmap(df_dual_ker.pivot("m", "nu", color), cbar=False, yticklabels=False, norm = LogNorm(),
                    annot=True, fmt=".2f", cmap="viridis", ax=axs[2])
        
        axs[0].set_title("Primal")
        axs[1].set_title("Dual (linear)")
        axs[2].set_title("Dual (kernel)")
        
    print(f"{color} heatmap")
    # legend execution time
    # set legend labels
    fig.colorbar(axs[1].collections[0], cax=axs[3])
    if save:
        fig.figure.savefig(f"plots/heatmap_{color}.pdf", bbox_inches='tight')
    plt.show()