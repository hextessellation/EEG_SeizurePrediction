import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import random
import os
import gc
import time
import sys
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                             roc_auc_score, f1_score)

# --- CONFIGURATION & CONSTANTS ---
# A longer training window to ensure convergence for difficult folds
SYNC_EPOCHS = 100
MIN_EPOCHS_BEFORE_STOP = 40
PATIENCE = 20
# A more balanced data signal to promote stability
UNDERSAMPLE_RATIO = 2.5 
BATCH_SIZE = 64
RANDOM_STATE = 42
FIXED_THRESHOLD = 0.5
MODEL_SAVE_DIR = "/kaggle/working/models/"
RESULTS_SAVE_PATH = "/kaggle/working/final_definitive_results.csv"

# --- HELPER CLASSES AND FUNCTIONS ---

def seed_everything(seed=RANDOM_STATE):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FocalLoss(nn.Module):
    """
    The one crucial tweak: a biased Focal Loss (alpha=0.75) to provide a stable 
    incentive for detecting the minority (seizure) class.
    """
    def __init__(self, alpha=0.65, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha).to(logits.device)
        focal_loss = alpha_t * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

class EEGSegmentDataset(Dataset):
    """Dataset class for loading EEG spectrogram segments."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spectro = np.load(row['file_path'], mmap_mode='r')[row['segment_idx']]
        return torch.tensor(spectro, dtype=torch.float32), torch.tensor(int(row['label']), dtype=torch.long)

class DeeperCNN_BiLSTM(nn.Module):
    """
    The main model architecture, using moderate dropout values from the original
    stable pipeline.
    """
    def __init__(self, num_eeg_channels=22, num_classes=2, 
                 conv_dropout=0.4, lstm_dropout=0.5, classifier_dropout=0.5):
        super(DeeperCNN_BiLSTM, self).__init__()
        self.conv_block1 = self._create_conv_block(num_eeg_channels, 32, conv_dropout)
        self.conv_block2 = self._create_conv_block(32, 64, conv_dropout)
        self.conv_block3 = self._create_conv_block(64, 128, conv_dropout)
        self.conv_block4 = self._create_conv_block(128, 256, conv_dropout, pool=False)
        
        final_conv_height = 112 // (2**3) 
        self.lstm_input_features = 256 * final_conv_height
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_features, hidden_size=256, num_layers=2, 
            batch_first=True, bidirectional=True, dropout=lstm_dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * 256, 128), nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, num_classes)
        )
    def _create_conv_block(self, in_channels, out_channels, dropout_rate, pool=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv_block1(x); x = self.conv_block2(x); x = self.conv_block3(x); x = self.conv_block4(x)
        x = x.permute(0, 3, 1, 2); x = x.reshape(x.size(0), x.size(1), -1)
        _, (h_n, _) = self.lstm(x)
        final_hidden_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        return self.classifier(final_hidden_state)

def get_loso_session_folds(main_df, session_map):
    """Creates training and testing folds based on a Leave-One-Session-Out strategy."""
    folds = []
    curated_preictal_files = list(session_map.keys())
    for held_out_preictal_file, held_out_interictal_buddies in session_map.items():
        held_out_session_files = [held_out_preictal_file] + held_out_interictal_buddies
        test_df = main_df[main_df['file_name'].isin(held_out_session_files)].reset_index(drop=True)
        
        train_preictal_filenames = [f for f in curated_preictal_files if f != held_out_preictal_file]
        train_preictal_df = main_df[main_df['file_name'].isin(train_preictal_filenames)]
        
        all_interictal_df = main_df[main_df['label'] == 0]
        train_interictal_df = all_interictal_df[~all_interictal_df['file_name'].isin(held_out_session_files)]
        
        train_df = pd.concat([train_preictal_df, train_interictal_df]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        
        if len(test_df) == 0 or len(test_df[test_df['label']==1]) == 0:
            print(f"Skipping fold for {held_out_preictal_file} due to no test samples or no positive labels in test set.")
            continue
            
        folds.append((train_df, test_df, held_out_preictal_file))
    return folds


class SeizurePredictionPipeline:
    def __init__(self, model_class, model_params, optimizer_params, device):
        self.model_class = model_class
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.device = device

    def train_model(self, train_df, val_df, model_path):
        model = self.model_class(**self.model_params).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), **self.optimizer_params)
        
        train_balanced_df = self._random_undersample(train_df, ratio=UNDERSAMPLE_RATIO)
        print(f"  Training data balanced. Seizure samples: {len(train_balanced_df[train_balanced_df['label']==1])}, Non-seizure samples: {len(train_balanced_df[train_balanced_df['label']==0])}")
        train_loader = DataLoader(EEGSegmentDataset(train_balanced_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(EEGSegmentDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SYNC_EPOCHS)
        criterion = FocalLoss().to(self.device)
        
        best_score, patience_counter = -1.0, 0
        print(f"  Starting training for up to {SYNC_EPOCHS} epochs...")
        for epoch in range(SYNC_EPOCHS):
            model.train(); epoch_start_time = time.time(); total_train_loss = 0
            
            for i, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item() * xb.size(0)
            
            scheduler.step()

            avg_train_loss = total_train_loss / len(train_loader.dataset)
            val_probs, val_labels, avg_val_loss = self._get_predictions_and_loss(model, val_loader, criterion)
            
            val_preds = (val_probs[:, 1] > FIXED_THRESHOLD).astype(int)
            val_auc = roc_auc_score(val_labels, val_probs[:, 1]) if len(np.unique(val_labels)) > 1 else 0.5
            val_f1_binary = f1_score(val_labels, val_preds, pos_label=1, zero_division=0)
            
            # The definitive saving metric: rewards both AUC and F1 for the seizure class.
            composite_score = 0.7 * val_auc + 0.3 * val_f1_binary
            sensitivity = recall_score(val_labels, val_preds, pos_label=1, zero_division=0)
            
            epoch_time = time.time() - epoch_start_time
            sys.stdout.write('\r' + ' ' * 80 + '\r'); sys.stdout.flush()
            print(f"  Epoch {epoch+1}/{SYNC_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1 (S): {val_f1_binary:.4f} | Sensitivity: {sensitivity:.4f} | Composite: {composite_score:.4f} | Time: {epoch_time:.1f}s")
            
            if composite_score > best_score:
                best_score = composite_score; patience_counter = 0
                torch.save(model.state_dict(), model_path)
                print(f"    -> Model saved to {model_path} (New best score: {best_score:.4f})")
            else:
                patience_counter += 1
            if epoch >= MIN_EPOCHS_BEFORE_STOP and patience_counter >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch+1}.")
                break
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        return model

    def _get_predictions_and_loss(self, model, data_loader, criterion):
        model.eval()
        all_probs, all_labels, total_val_loss = [], [], 0
        with torch.no_grad():
            for xb, yb in data_loader:
                xb_dev, yb_dev = xb.to(self.device), yb.to(self.device)
                logits = model(xb_dev)
                loss = criterion(logits, yb_dev)
                total_val_loss += loss.item() * xb.size(0)
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                all_labels.append(yb.cpu().numpy())
        avg_loss = total_val_loss / len(data_loader.dataset) if len(data_loader.dataset) > 0 else 0
        return np.vstack(all_probs), np.concatenate(all_labels), avg_loss
    
    def _random_undersample(self, df, ratio, random_state=RANDOM_STATE):
        minority = df[df['label'] == 1]
        majority = df[df['label'] == 0]
        n_majority = int(len(minority) * ratio)
        if n_majority > len(majority): n_majority = len(majority)
        majority_downsampled = majority.sample(n=n_majority, random_state=random_state)
        return pd.concat([minority, majority_downsampled]).sample(frac=1, random_state=random_state)

def evaluate_with_threshold(y_true, y_probs, threshold):
    """Calculates key metrics for a given threshold."""
    y_pred = (y_probs > threshold).astype(int)
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    interictal_segments = np.sum(y_true == 0)
    
    interictal_hours = (interictal_segments * 30.0) / 3600.0
    fph = false_positives / interictal_hours if interictal_hours > 0 else 0.0
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_score_binary': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'sensitivity': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'fph': fph,
        'auc_roc': roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5,
        'threshold': threshold
    }

if __name__ == '__main__':
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # NOTE: Assume `cleaned_df` and `session_map` are pre-loaded.
    
    folds = get_loso_session_folds(cleaned_df, session_map)
    all_final_metrics = []
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Returning to the stable, moderately-regularized model parameters
    model_params = {'num_eeg_channels': 22, 'num_classes': 2, 'conv_dropout': 0.4, 'lstm_dropout': 0.5, 'classifier_dropout': 0.5}
    # Using a conservative learning rate to ensure stable learning
    optimizer_params = {'lr': 1e-3, 'momentum': 0.9, 'weight_decay': 5e-4}
    
    for i, (train_df, test_df, held_out_file) in enumerate(folds):
        fold_id = i + 1
        print(f"\n{'='*50}\nSTARTING FOLD {fold_id}/{len(folds)}\n{'='*50}")
        print(f"Held-out session for validation: {held_out_file}")
        
        pipeline = SeizurePredictionPipeline(DeeperCNN_BiLSTM, model_params, optimizer_params, device)
        model_path = os.path.join(MODEL_SAVE_DIR, f"loso_model_fold_{fold_id}.pt")
        
        model = pipeline.train_model(train_df, test_df, model_path)
        
        print(f"\n--- Evaluating Fold {fold_id} with best model... ---")
        fold_probs, fold_labels, _ = pipeline._get_predictions_and_loss(
            model, DataLoader(EEGSegmentDataset(test_df), batch_size=BATCH_SIZE*2), nn.CrossEntropyLoss()
        )
        
        fold_metrics = evaluate_with_threshold(fold_labels, fold_probs[:, 1], FIXED_THRESHOLD)
        print(f"  -> Fold Metrics (Threshold = {FIXED_THRESHOLD}):")
        print(f"     Accuracy:        {fold_metrics['accuracy']:.4f}")
        print(f"     F1-Score (Macro):{fold_metrics['f1_score_macro']:.4f}")
        print(f"     F1-Score (S):    {fold_metrics['f1_score_binary']:.4f}")
        print(f"     Sensitivity:     {fold_metrics['sensitivity']:.4f}")
        print(f"     Precision:       {fold_metrics['precision']:.4f}")
        print(f"     FPH:             {fold_metrics['fph']:.4f}")
        print(f"     AUC:             {fold_metrics['auc_roc']:.4f}")
        
        fold_metrics['fold'] = fold_id
        fold_metrics['held_out_file'] = held_out_file
        all_final_metrics.append(fold_metrics)

        del model, pipeline; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_final_metrics)
    
    print(f"\n{'='*50}\nFINAL RESULTS\n{'='*50}")
    
    print("\n--- FINAL FOLD-BY-FOLD METRICS (USING FIXED THRESHOLD) ---")
    display_cols = ['held_out_file', 'accuracy', 'sensitivity', 'precision', 'f1_score_binary', 'f1_score_macro', 'auc_roc', 'fph']
    print(results_df[display_cols].to_string(index=False))
    
    print(f"\n\n--- FINAL AGGREGATED RESULTS (Mean) ---")
    mean_metrics = results_df[display_cols].select_dtypes(include=np.number).mean()
    print(mean_metrics)
    
    results_df.to_csv(RESULTS_SAVE_PATH, index=False)
    print(f"\nFull results saved to {RESULTS_SAVE_PATH}")
