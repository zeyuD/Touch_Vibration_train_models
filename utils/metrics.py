import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn

def plot_results(metrics_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))
    for m in metrics_list:
        plt.plot(m['fpr'], m['tpr'], lw=2, label=f"{m['target_key']} (AUC={m['auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))

    x = np.arange(len(metrics_list))
    keys = [m['target_key'] for m in metrics_list]
    fars = [m['far'] for m in metrics_list]
    frrs = [m['frr'] for m in metrics_list]
    accs = [m['accuracy'] / 100 for m in metrics_list]
    width = 0.25

    plt.figure(figsize=(14, 6))
    plt.bar(x - width, fars, width, label='FAR')
    plt.bar(x, frrs, width, label='FRR')
    plt.bar(x + width, accs, width, label='Acc/100')
    plt.xticks(x, keys, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'far_frr_comparison.png'))

    df = pd.DataFrame({
        'Target': keys,
        'Accuracy (%)': [m['accuracy'] for m in metrics_list],
        'AUC': [m['auc'] for m in metrics_list],
        'FAR': fars,
        'FRR': frrs
    })
    df.to_csv(os.path.join(save_dir, 'performance_summary.csv'), index=False)
    print(f"Saved plots and summary to: {save_dir}")
    return df
