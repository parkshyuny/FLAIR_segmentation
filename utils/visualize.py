import matplotlib.pyplot as plt
import numpy as np

def sample_images(y_pred, y_true, subplots=(2,5), figsize=(22,8), save=False):
    plt.figure(figsize=figsize)

    # Predicted masks
    i = 0
    for mask in y_pred:
        if i == 5: 
            break

        plt.subplot(subplots[0], subplots[1], i+1)
        plt.imshow(mask.transpose(1, 2, 0), cmap='gray')
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.axis('off')
        i += 1

    # Ground truth masks
    for mask in y_true:
        if i == 10:
            break

        plt.subplot(subplots[0], subplots[1], i+1)
        plt.imshow(mask.transpose(1, 2, 0), cmap='gray')
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.axis('off')
        i += 1

    plt.tight_layout()
    plt.show()