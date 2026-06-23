import matplotlib.pyplot as plt
import numpy as np

def sample_images(y_pred, y_true, step, subplots=(2,5), figsize=(22,8), save=False):
    plt.figure(figsize=figsize)

    for i, image in enumerate(y_true):
        plt.subplot(subplots[0], subplots[1], i+1)
        plt.imshow(image.permute(1, 2, 0))
        plt.imshow(y_pred[i].permute(1, 2, 0), alpha=0.3, cmap='red')
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"step_{step}_prediction.png")
    plt.show()