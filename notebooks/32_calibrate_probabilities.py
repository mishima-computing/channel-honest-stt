import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import IsotonicRegression

def plot_calibration_curve(y_true, y_prob, output_path, title):
    bins = np.linspace(0.0, 1.0, 11)
    prob_bins = np.digitize(y_prob, bins) - 1
    prob_bins = np.clip(prob_bins, 0, 9)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = []
    counts = []
    
    for i in range(10):
        idx = (prob_bins == i)
        if np.sum(idx) > 0:
            accuracies.append(np.mean(y_true[idx]))
        else:
            accuracies.append(np.nan)
        counts.append(np.sum(idx))
        
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(bin_centers, accuracies, 'o-', color='b', label='Actual Accuracy')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax1.set_xlabel('Calibrated Confidence')
    ax1.set_ylabel('Actual Accuracy')
    ax1.set_ylim([0, 1.05])
    ax1.set_xlim([0, 1.0])
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, counts, width=0.08, alpha=0.3, color='g', label='Count')
    ax2.set_ylabel('Number of Samples')
    
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_distribution(y_prob, output_path, title):
    plt.figure(figsize=(8, 6))
    plt.hist(y_prob, bins=np.linspace(0, 1, 21), edgecolor='black', alpha=0.7)
    plt.xlabel('Calibrated Confidence')
    plt.ylabel('Count')
    plt.title(title)
    
    high_conf = np.sum(y_prob >= 0.7)
    total = len(y_prob)
    plt.axvline(0.7, color='r', linestyle='dashed', linewidth=2)
    plt.text(0.72, plt.gca().get_ylim()[1]*0.9, f'>=0.7:\n{high_conf}/{total}\n({high_conf/total*100:.1f}%)', color='r')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_tradeoff_curve(y_true, y_prob, output_path, title):
    thresholds = np.linspace(0.0, 1.0, 101)
    conf_error_rates = []
    flow_volumes = []
    
    total = len(y_true)
    for t in thresholds:
        high_conf_idx = (y_prob >= t)
        low_conf_idx = (y_prob < t)
        
        flow_volume = np.sum(low_conf_idx) / total
        if np.sum(high_conf_idx) > 0:
            err_rate = 1.0 - np.mean(y_true[high_conf_idx])
        else:
            err_rate = 0.0
            
        conf_error_rates.append(err_rate)
        flow_volumes.append(flow_volume)
        
    plt.figure(figsize=(8, 6))
    plt.plot(flow_volumes, conf_error_rates, 'r-', linewidth=2)
    plt.xlabel('Confirmation Flow Volume (% of tokens)')
    plt.ylabel('Confident Error Rate')
    plt.title(title)
    plt.grid(True)
    
    for t_val in [0.5, 0.7, 0.8, 0.9]:
        idx = np.argmin(np.abs(thresholds - t_val))
        plt.plot(flow_volumes[idx], conf_error_rates[idx], 'ko')
        plt.annotate(f'T={t_val:.1f}', (flow_volumes[idx], conf_error_rates[idx]), 
                     xytext=(10, 10), textcoords='offset points')
                     
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    base_dir = os.path.dirname(__file__)
    
    for snr in ['0dB', '10dB']:
        csv_path = os.path.join(base_dir, f'e2e_predictions_{snr}.csv')
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        df['pred_label'] = df['pred_label'].astype(str).str.replace(' ', '_')
        df['true_label'] = df['true_label'].astype(str).str.replace(' ', '_')
        
        # Binary target: 1 if correct, 0 if wrong
        df['is_correct'] = (df['true_label'] == df['pred_label']).astype(int)
        
        # Split sentences into train/test for held-out calibration
        sentences = df['sentence_id'].unique()
        np.random.seed(42)
        np.random.shuffle(sentences)
        
        train_sents = set(sentences[:len(sentences)//2])
        test_sents = set(sentences[len(sentences)//2:])
        
        df_train = df[df['sentence_id'].isin(train_sents)].copy()
        df_test = df[df['sentence_id'].isin(test_sents)].copy()
        
        print(f"\n--- {snr} Calibration ---")
        print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")
        
        # Fit Isotonic Regression on train
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(df_train['max_prob'], df_train['is_correct'])
        
        # Predict calibrated probs on test
        df_test['calib_prob'] = ir.predict(df_test['max_prob'])
        
        y_true_test = df_test['is_correct'].values
        y_prob_calib = df_test['calib_prob'].values
        
        # Plot held-out calibration curve
        calib_path = os.path.join(base_dir, f'{snr}_calibrated_calibration.png')
        plot_calibration_curve(y_true_test, y_prob_calib, calib_path, f'Calibrated Curve (Held-out Test, {snr})')
        
        # Plot distribution of calibrated probs
        dist_path = os.path.join(base_dir, f'{snr}_calibrated_distribution.png')
        plot_distribution(y_prob_calib, dist_path, f'Distribution of Calibrated Confidence ({snr})')
        
        # Plot new tradeoff
        tradeoff_path = os.path.join(base_dir, f'{snr}_calibrated_tradeoff.png')
        plot_tradeoff_curve(y_true_test, y_prob_calib, tradeoff_path, f'Calibrated Tradeoff ({snr})')
        
        print(f"[{snr}] Generated calibrated plots.")
        
        # Count how many >= 0.7
        high_conf = np.sum(y_prob_calib >= 0.7)
        total = len(y_prob_calib)
        print(f"[{snr}] Tags with calibrated confidence >= 0.7: {high_conf}/{total} ({high_conf/total*100:.1f}%)")

if __name__ == '__main__':
    main()
