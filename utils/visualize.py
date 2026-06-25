import matplotlib.pyplot as plt
import numpy as np

def visualize_preds(y_pred, y_true, epoch, subplots=(2,5), figsize=(22,8), save=False):
    plt.figure(figsize=figsize)

    y_pred = y_pred.detach()
    y_true = y_true.detach()

    loc = 1
    for i, mask in enumerate(y_pred):
        if i == 5: 
            break
        plt.subplot(subplots[0], subplots[1], loc)
        plt.imshow(np.transpose(mask, (2, 1, 0)))
        loc += 1

        plt.subplots_adjust(wspace=None, hspace=None)
        plt.axis('off')

    for i, mask in enumerate(y_true):
        if i == 5: 
            break
        plt.subplot(subplots[0], subplots[1], loc)
        plt.imshow(np.transpose(mask, (2, 1, 0)))
        loc += 1

        plt.subplots_adjust(wspace=None, hspace=None)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(f"predicted_masks/epoch_{epoch}_preds.png")
    plt.show()