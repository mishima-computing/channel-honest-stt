import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_calibration_curve(df, output_path):
    bins = np.linspace(0.0, 1.0, 11)
    df['prob_bin'] = pd.cut(df['max_prob'], bins=bins, labels=False, include_lowest=True)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = []
    counts = []
    
    for i in range(10):
        bin_df = df[df['prob_bin'] == i]
        if len(bin_df) > 0:
            acc = (bin_df['true_label'] == bin_df['pred_label']).mean()
            accuracies.append(acc)
        else:
            accuracies.append(np.nan)
        counts.append(len(bin_df))
        
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    ax1.plot(bin_centers, accuracies, 'o-', color='b', label='Actual Accuracy')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax1.set_xlabel('Confidence (Max Probability)')
    ax1.set_ylabel('Actual Accuracy')
    ax1.set_ylim([0, 1.05])
    ax1.set_xlim([0, 1.0])
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, counts, width=0.08, alpha=0.3, color='g', label='Count')
    ax2.set_ylabel('Number of Samples')
    
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
    plt.title('Confidence Calibration Curve (MUSAN Babble 0dB)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_tradeoff_curve(df, output_path):
    thresholds = np.linspace(0.0, 1.0, 101)
    conf_error_rates = []
    flow_volumes = []
    
    total = len(df)
    for t in thresholds:
        high_conf = df[df['max_prob'] >= t]
        low_conf = df[df['max_prob'] < t]
        
        flow_volume = len(low_conf) / total
        if len(high_conf) > 0:
            err_rate = (high_conf['true_label'] != high_conf['pred_label']).mean()
        else:
            err_rate = 0.0
            
        conf_error_rates.append(err_rate)
        flow_volumes.append(flow_volume)
        
    plt.figure(figsize=(8, 6))
    plt.plot(flow_volumes, conf_error_rates, 'r-', linewidth=2)
    plt.xlabel('Confirmation Flow Volume (% of tokens)')
    plt.ylabel('Confident Error Rate')
    plt.title('Tradeoff: Confirmation Flow vs Confident Error Rate')
    plt.grid(True)
    
    # Annotate some thresholds
    for t_val in [0.5, 0.7, 0.8, 0.9]:
        idx = np.argmin(np.abs(thresholds - t_val))
        plt.plot(flow_volumes[idx], conf_error_rates[idx], 'ko')
        plt.annotate(f'T={t_val:.1f}', (flow_volumes[idx], conf_error_rates[idx]), 
                     xytext=(10, 10), textcoords='offset points')
                     
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def plot_uv_fricative_false_positives(df, output_path):
    # Only consonants
    df_cons = df[~df['is_vowel']]
    
    # False positives for Unvoiced Fricative
    uvf_fp = df_cons[(df_cons['pred_label'] == 'Unvoiced_Fricative') & (df_cons['true_label'] != 'Unvoiced_Fricative')]
    
    plt.figure(figsize=(8, 6))
    plt.hist(uvf_fp['max_prob'], bins=np.linspace(0, 1, 21), edgecolor='black', alpha=0.7)
    plt.axvline(uvf_fp['max_prob'].mean(), color='r', linestyle='dashed', linewidth=2, label=f"Mean: {uvf_fp['max_prob'].mean():.2f}")
    
    plt.xlabel('Confidence (Max Probability)')
    plt.ylabel('Count')
    plt.title('Confidence Distribution of <Unvoiced_Fricative> False Positives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"\n[Unvoiced Fricative False Positives Analysis]")
    print(f"Total false positives: {len(uvf_fp)}")
    print(f"Mean confidence: {uvf_fp['max_prob'].mean():.3f}")
    high_conf_fp = len(uvf_fp[uvf_fp['max_prob'] >= 0.7])
    print(f"False positives with >= 0.7 confidence: {high_conf_fp} ({high_conf_fp/len(uvf_fp)*100:.1f}%)")
    
    # Also look at what they actually were
    print("\nTrue labels of these false positives:")
    print(uvf_fp['true_label'].value_counts())

def main():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, 'e2e_predictions_0dB.csv')
    
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    # Normalize labels: replace spaces with underscores to match true_label format
    df['pred_label'] = df['pred_label'].astype(str).str.replace(' ', '_')
    df['true_label'] = df['true_label'].astype(str).str.replace(' ', '_')
    print(f"Loaded {len(df)} predictions.")
    
    calib_path = os.path.join(base_dir, 'confidence_calibration.png')
    plot_calibration_curve(df, calib_path)
    print(f"Saved calibration curve to {calib_path}")
    
    tradeoff_path = os.path.join(base_dir, 'confidence_tradeoff.png')
    plot_tradeoff_curve(df, tradeoff_path)
    print(f"Saved tradeoff curve to {tradeoff_path}")
    
    fp_path = os.path.join(base_dir, 'uv_fricative_fp_dist.png')
    plot_uv_fricative_false_positives(df, fp_path)
    print(f"Saved UV Fricative FP distribution to {fp_path}")

if __name__ == '__main__':
    main()
