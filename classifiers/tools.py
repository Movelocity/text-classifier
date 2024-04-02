import matplotlib.pyplot as plt
import numpy as np


def show_diff(probs, labels):
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

def plot_training_history(
        figsize, xs, data=[], 
        labels=['loss', 'top1', 'top5'], 
        colors=['blue', 'green', 'red']
    ):
    fig, ax = plt.subplots(figsize=figsize)

    for index in range(len(data)):
        ax.plot(xs, data[index], label=labels[index], color=colors[index], alpha=0.4)
        interval = len(xs)//8
        for i, (_x, _y) in enumerate(zip(xs, data[index])):  
            if i % interval == 0:   # annotate for every nth point
                ax.annotate(
                    f'{_y:.2f}', (_x, _y), 
                    textcoords="offset points", 
                    xytext=(0,10), ha='center', 
                    color=colors[index]
                )

    ax.set_xlabel("step")
    ax.set_title('finetune bert')
    ax.legend()
    plt.savefit("training_history.jpg")
    plt.show()