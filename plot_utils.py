import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os 

colorlist = list(colors.TABLEAU_COLORS)

def double_heatmap_plot(probs, labels):
    label_space = np.zeros(probs.shape)
    for i, v in enumerate(labels):
        label_space[i][v] = 1

    plt.figure(figsize=(6, 4))
    plt.subplot(121)
    plt.imshow(probs)
    plt.title('prediction')
    plt.subplot(122)
    plt.imshow(label_space)
    plt.title('target')
    plt.show()


def multiline_plot(xs, ys:dict, figsize=(6, 4), title='', xlabel='', ylabel='', save_name=''):
    fig, ax = plt.subplots(figsize=figsize)
    for idx, (name, values) in enumerate(ys.items()):
        ax.plot(xs, values, label=name, color=colorlist[idx], alpha=0.4)
        interval = len(xs)//9 if len(xs) > 9 else 1
        for i, (_x, _y) in enumerate(zip(xs, values)):
            if i % interval == 0:   # annotate for every nth point
                ax.annotate(
                    f'{_y:.2f}', (_x, _y), 
                    textcoords="offset points", 
                    xytext=(0,10), ha='center', 
                    color=colorlist[idx]
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Expand the y-axis limits
    y_min, y_max = ax.get_ylim()
    expand_ratio = 0.15
    expand_value = (y_max - y_min) * expand_ratio
    ax.set_ylim(y_min - expand_value, y_max + expand_value)

    ax.legend()
    if save_name and save_name[-4:] in ['.jpg', '.png']:
        plt.savefit(save_name)
    plt.show()