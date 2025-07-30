import os
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
PATIENT_ID = 'Patient-1'
SEGMENTS_DIR = '/kaggle/input/patient-1-30seconds/Patient_1'
ANNOTATION_CSV = '/kaggle/input/patient-1-30seconds/chb1_segments.csv'

# 1. Load annotation DataFrame
ann_df = pd.read_csv(ANNOTATION_CSV)

# 2. Prepare a list to collect data
data = []

# 3. Loop over all segmented files for this patient
for seg_file in sorted(os.listdir(SEGMENTS_DIR)):
    if not (seg_file.endswith('.npy')):
        continue
    seg_path = os.path.join(SEGMENTS_DIR, seg_file)
    segments = np.load(seg_path, mmap_mode='r')  # don't load into memory

    # Adjust file name for looking up in annotation DataFrame
    base_seg_file = seg_file.replace('_stft_30sec.npy', '_segmented.npy')
    file_ann = ann_df[ann_df['filename'] == base_seg_file]

    for _, row in file_ann.iterrows():
        seg_idx = int(row['segment'])
        label = int(row['segment_label'])
        if seg_idx >= len(segments):
            continue
        data.append({
            'filename': seg_file,
            'segment_idx': seg_idx,
            'label': label,
            'file_path': seg_path  # store path only
        })

# 4. Convert to DataFrame
df_segments = pd.DataFrame(data)

print(f"Stored {len(df_segments)} labeled segment references for {PATIENT_ID}.")

import pandas as pd
import numpy as np # Ensure numpy is imported for rng in get_loso_folds

def process_segments(df_raw, preictal_threshold=200):
    """
    Filters the raw DataFrame to include only 'interictal' (0) and 'preictal' (1) segments.
    Identifies files that have enough preictal segments to serve as LOSO validation/test folds.

    Args:
        df_raw (pd.DataFrame): DataFrame with raw segment labels (0, 1, 2, 3, 4).
                               Expected columns: 'filename', 'label', etc.
        preictal_threshold (int): Minimum number of preictal segments a file must have
                                  to be considered a candidate for a LOSO validation/test fold.

    Returns:
        tuple:
            - cleaned_df (pd.DataFrame): DataFrame containing only interictal (0) and preictal (1) segments.
                                         This is the full pool of data for the patient.
            - loso_candidate_files (list): List of filenames that qualify to be held out as a
                                           validation/test set in a LOSO fold (i.e., contain >= preictal_threshold preictal segments).
    """
    # Step 1: Remove unwanted labels (ictal=2, postictal=3, grey=4)
    # This creates the universal pool of usable segments (interictal and preictal) for the patient.
    cleaned_df = df_raw[~df_raw['label'].isin([2, 3, 4])].reset_index(drop=True)

    # Step 2: Identify files that have enough preictal (label=1) segments
    # These are the candidates for being the held-out file in a LOSO fold.
    preictal_df_only = cleaned_df[cleaned_df['label'] == 1]
    preictal_counts = preictal_df_only['filename'].value_counts()

    # Files that meet the threshold to be a LOSO validation/test fold
    loso_candidate_files = preictal_counts[preictal_counts >= preictal_threshold].index.tolist()

    # IMPORTANT: The 'cleaned_df' is your full pool.
    # The 'loso_candidate_files' only dictates WHICH files can be held out as a test fold.
    # Segments from files NOT in 'loso_candidate_files' will always be part of the training set
    # for any given fold.

    print(f"Total usable segments (interictal/preictal): {len(cleaned_df)}")
    print(f"Files qualifying as LOSO validation/test candidates (>= {preictal_threshold} preictal segments): {len(loso_candidate_files)}")
    if not loso_candidate_files:
        print("WARNING: No files met the preictal threshold to form LOSO folds. Check your data or threshold.")

    return cleaned_df, loso_candidate_files

cleaned_df, loso_candidate_files = process_segments(df_segments, preictal_threshold=40)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import gc
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, brier_score_loss, f1_score
import time
import pandas as pd
import os
import math # Needed for CosineAnnealingLR T_max calculation
from torchvision import models # Needed for torchvision.models.vit_b_16

# --- EEGHybridViT Model Definition (Consolidated from your previous code) ---
class EEGHybridViT(nn.Module):
    def __init__(self, num_eeg_channels=22, num_classes=2):
        super().__init__()
        # Multi-layer convolutional feature extractor (adapts input channels to 3)
        # Assuming input spectrograms are approx 224x224 and conv_layers maintains spatial dims.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_eeg_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # Reduce to 3 channels
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        # Pretrained ViT (vit_b_16 expects 224x224 input)
        self.vit = models.vit_b_16(pretrained=True)
        
        # Classifier head: 2 dense layers, replacing the original ViT head
        # Dropout adjusted here to 0.3 for consistency with previous aggressive settings
        self.vit.heads.head = nn.Sequential(
            nn.Linear(self.vit.heads.head.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3), # Using 0.3 for classifier dropout
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x is (batch, num_eeg_channels, H, W) e.g., (batch, 22, 257, 225) after STFT
        x = self.conv_layers(x)  # Output: (batch, 3, H, W) where H, W are maintained due to padding=1 and no pooling.
        
        # Interpolate to 224x224 for the pretrained ViT input
        x = nn.functional.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        return self.vit(x)

# --- END EEGHybridViT Model Definition ---


# ---- Simple Dataset for Precomputed Spectrograms (kept as provided) ----
class EEGSegmentDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['file_path']
        segment_idx = row['segment_idx']
        label = int(row['label'])
        spectro = np.load(file_path, mmap_mode='r')[segment_idx]
        spectro_tensor = torch.tensor(spectro, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return spectro_tensor, label_tensor

# --- Random undersampling function (kept as provided) ---
def random_undersample(df, label_col='label', majority_class=0, minority_class=1, random_state=42, ratio=1.0):
    df_majority = df[df[label_col] == majority_class]
    df_minority = df[df[label_col] == minority_class]
    n_majority = int(len(df_minority) * ratio)
    df_majority_downsampled = df_majority.sample(n=min(n_majority, len(df_majority)), random_state=random_state)
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df_balanced

# --- augment_batch function (as provided) ---
def augment_batch(batch_spectrograms, p_augment=0.5, 
                  mag_scale_range=(0.9, 1.1),
                  freq_mask_max_ratio=0.1, time_mask_max_ratio=0.1,
                  device='cuda'):
    augmented_batch = batch_spectrograms.clone().to(device)
    batch_size, num_channels, H, W = augmented_batch.shape
    for i in range(batch_size):
        if random.random() < p_augment:
            spectrogram = augmented_batch[i]
            scale_factor = random.uniform(mag_scale_range[0], mag_scale_range[1])
            spectrogram = spectrogram * scale_factor
            if random.random() < 0.5:
                num_masked_freq_bins = int(H * random.uniform(0.0, freq_mask_max_ratio))
                if num_masked_freq_bins > 0:
                    start_freq_bin = random.randint(0, H - num_masked_freq_bins)
                    spectrogram[:, start_freq_bin : start_freq_bin + num_masked_freq_bins, :] = 0.0
            if random.random() < 0.5:
                num_masked_time_bins = int(W * random.uniform(0.0, time_mask_max_ratio))
                if num_masked_time_bins > 0:
                    start_time_bin = random.randint(0, W - num_masked_time_bins)
                    spectrogram[:, :, start_time_bin : start_time_bin + num_masked_time_bins] = 0.0
            augmented_batch[i] = spectrogram
    return augmented_batch

# --- CORRECTED get_loso_folds function (as provided) ---
def get_loso_folds(cleaned_df, loso_candidate_files, interictal_val_ratio=0.2, random_state=42):
    folds = []
    rng = np.random.RandomState(random_state)
    all_preictal_segments = cleaned_df[cleaned_df['label'] == 1]
    all_interictal_segments = cleaned_df[cleaned_df['label'] == 0]
    all_interictal_filenames = all_interictal_segments['filename'].unique().tolist()
    rng.shuffle(all_interictal_filenames)
    num_interictal_val_files = max(1, int(len(all_interictal_filenames) * interictal_val_ratio))
    global_interictal_val_filenames = all_interictal_filenames[:num_interictal_val_files]
    global_interictal_train_filenames = all_interictal_filenames[num_interictal_val_files:]
    shuffled_loso_candidate_files = list(loso_candidate_files)
    rng.shuffle(shuffled_loso_candidate_files)
    if not shuffled_loso_candidate_files:
        print("No LOSO candidate files found. Cannot create preictal-focused folds.")
        return []
    for held_out_filename_preictal in shuffled_loso_candidate_files:
        val_preictal_part = all_preictal_segments[all_preictal_segments['filename'] == held_out_filename_preictal]
        val_interictal_part = all_interictal_segments[
            all_interictal_segments['filename'].isin(global_interictal_val_filenames)
        ]
        val_df = pd.concat([val_preictal_part, val_interictal_part]).reset_index(drop=True)
        train_preictal_part = all_preictal_segments[
            all_preictal_segments['filename'] != held_out_filename_preictal
        ]
        train_interictal_part = all_interictal_segments[
            all_interictal_segments['filename'].isin(global_interictal_train_filenames)
        ]
        train_df = pd.concat([train_preictal_part, train_interictal_part]).reset_index(drop=True)
        train_preictal_count = len(train_df[train_df['label'] == 1])
        train_interictal_count = len(train_df[train_df['label'] == 0])
        val_preictal_count = len(val_df[val_df['label'] == 1])
        val_interictal_count = len(val_df[val_df['label'] == 0])
        print(f"\n--- Fold for test file: {held_out_filename_preictal} ---")
        print(f"Train Segments: {len(train_df)} (Preictal: {train_preictal_count}, Interictal: {train_interictal_count})")
        print(f"Validation Segments: {len(val_df)} (Preictal: {val_preictal_count}, Interictal: {val_interictal_count})")
        if val_preictal_count == 0 or val_interictal_count == 0:
            print(f"WARNING: Validation set for this fold is missing a class! Preictal: {val_preictal_count}, Interictal: {val_interictal_count}. Some metrics might be NaN or misleading.")
        folds.append((train_df, val_df, held_out_filename_preictal))
    return folds

# --- Main training function (with all corrections and consolidations) ---
def train_patient_model(
    train_df,
    val_df,
    model, # This model instance is now passed in and expected to be newly initialized per fold
    batch_size=8,
    num_epochs=100,
    patience=15, # Consolidated patience
    lr=5e-5,     # Consolidated learning rate
    undersample=True,
    undersample_ratio=1.0, # Consolidated undersample ratio to 1.0 for 1:1 balance
    augmenter=None, # Placeholder, augment_batch is called directly
    accumulation_steps=4, # Assuming this is passed from main loop
    model_path="/kaggle/working/best_model.pt" # This path MUST be unique per fold
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if undersample:
        train_df = random_undersample(
            train_df,
            label_col='label',
            majority_class=0,
            minority_class=1,
            random_state=42,
            ratio=undersample_ratio # Use consolidated ratio
        )
        print(f"Train samples after undersampling: {len(train_df)}")
        print("Train class counts after undersampling:", train_df['label'].value_counts().to_dict())
            
    print(f"Train_df labels: {train_df['label'].unique()}, Val_df labels: {val_df['label'].unique()}")
    
    if len(train_df) == 0:
        print("WARNING: Training DataFrame is empty. Skipping fold evaluation.")
        return {'best_val_f1': float('nan')}
    if len(val_df) == 0:
        print("WARNING: Validation DataFrame is empty. All validation metrics will be NaN.")

    train_dataset = EEGSegmentDataset(train_df)
    val_dataset = EEGSegmentDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = True

    num_classes = 2
    class_counts = train_df['label'].value_counts().reindex(range(num_classes), fill_value=0)
    safe_class_counts = class_counts.replace(0, 1e-6)
    
    weights = 1.0 / safe_class_counts # Using inverse frequency (no sqrt/boost) as in high-acc code
    weights = weights / weights.sum() # Normalize to sum to 1
    class_weights_tensor = torch.tensor(weights.values, dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01) # Consolidated weight_decay
    
    # --- Cosine Annealing LR Scheduler ---
    T_max_scheduler = num_epochs * (len(train_loader) // accumulation_steps) # Number of optimizer steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max_scheduler, eta_min=1e-6)
    # -------------------------------------
    
    grad_clip = 1.0

    print("Class weights:", class_weights_tensor.cpu().numpy().tolist())
    print(f"DEBUG: Accumulation steps received: {accumulation_steps}") # Keep this debug print

    best_f1 = -1.0
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        
        optimizer.zero_grad() # Zero gradients at the start of each accumulation cycle / epoch

        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            
            # Apply Augmentation
            xb = augment_batch(xb, p_augment=0.95, # Aggressive p_augment
                               mag_scale_range=(0.7, 1.3),
                               freq_mask_max_ratio=0.3, # Aggressive masking
                               time_mask_max_ratio=0.3, # Aggressive masking
                               device=device)
            
            assert not torch.isnan(xb).any() and not torch.isinf(xb).any(), "NaN/Inf in xb"
            
            logits = model(xb)
            loss = criterion(logits, yb)
            
            loss = loss / accumulation_steps # Scale loss
            loss.backward() # Accumulate gradients
            
            if (i + 1) % accumulation_steps == 0: # Execute optimizer step only after accumulation
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # Clip accumulated gradients
                optimizer.step() # Update weights
                scheduler.step() # Call scheduler.step() after optimizer.step()
                optimizer.zero_grad() # Clear gradients for the next accumulation cycle
            
            train_loss += loss.item() * xb.size(0) * accumulation_steps
            
            del xb, yb, logits, loss
            torch.cuda.empty_cache()
            
            elapsed = time.time() - start_time
            batches_done = i + 1
            batches_left = len(train_loader) - batches_done
            time_per_batch = elapsed / batches_done
            est_time_left = time_per_batch * batches_left
            print(f"Epoch {epoch+1}/{num_epochs} | Batch {batches_done}/{len(train_loader)} - Estimated time left: {est_time_left:.1f} seconds", end='\r')
        
        # --- Handle remaining gradients if the last batch was not a multiple of accumulation_steps ---
        if (i + 1) % accumulation_steps != 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step() # Call scheduler.step() for the final step
            optimizer.zero_grad()
        # ---------------------------------------------------------------------------------------

        train_loss /= len(train_loader.dataset)
        print() # For a clean new line after the batch progress output
        
        # ... (rest of validation and early stopping logic) ...
        model.eval()
        val_loss = 0
        all_logits, all_labels = [], []
        with torch.no_grad():
            if len(val_loader.dataset) > 0:
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)
                    all_logits.append(logits.cpu())
                    all_labels.append(yb.cpu())
                    del xb, yb, logits, loss
                    torch.cuda.empty_cache()
                val_loss /= len(val_loader.dataset)
            else:
                val_loss = float('nan')
                
        if len(all_labels) == 0:
            print("WARNING: Validation set was entirely empty. All metrics will be NaN.")
            val_f1 = val_acc = brier = val_auc = float('nan')
            val_recall = val_precision = np.array([float('nan'), float('nan')])
        else:
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)
            probs = torch.softmax(all_logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            unique_labels_val = all_labels.unique()
            if len(unique_labels_val) < 2:
                print(f"WARNING: Validation set only contains {len(unique_labels_val)} unique label(s). Some metrics might be NaN or misleading.")
                val_f1 = val_acc = brier = val_auc = float('nan')
                val_recall = val_precision = np.array([float('nan'), float('nan')])
            else:
                val_f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
                val_acc = accuracy_score(all_labels, preds)
                val_recall = recall_score(all_labels, preds, average=None, zero_division=0)
                val_precision = precision_score(all_labels, preds, average=None, zero_division=0)
                
                brier = np.mean([brier_score_loss((all_labels.numpy() == c).astype(int), probs[:, c].numpy()) for c in range(probs.shape[1])])
                try:
                    val_auc = roc_auc_score(all_labels, probs[:, 1].numpy())
                except ValueError as e:
                    print(f"WARNING: Could not calculate AUC: {e}. Likely only one class predicted or present in relevant labels.")
                    val_auc = float('nan')

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val Recall: {val_recall} | Val Precision: {val_precision} | "
              f"Brier: {brier:.4f} | Val AUC: {val_auc:.4f}")

        # Note: Cosine Annealing does not typically require scheduler.step(val_loss)
        # However, for early stopping (patience_counter), you still track val_loss.
        if not np.isnan(val_loss) and val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_f1 = val_f1
            patience_counter = 0
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved with improved Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter > patience:
                print("Early stopping triggered.")
                break

    print("Training complete for this fold.")
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"WARNING: Best model not saved at {model_path} during training. Returning metrics from last epoch.")
        
    metrics = {'best_val_f1': best_f1,
               'final_val_loss': val_loss,
               'final_val_acc': val_acc,
               'final_val_recall': val_recall.tolist(),
               'final_val_precision': val_precision.tolist(),
               'final_brier': brier,
               'final_val_auc': val_auc}
    
    del model, train_loader, val_loader, train_dataset, val_dataset
    torch.cuda.empty_cache()
    gc.collect()
    
    return metrics


# --- Main execution loop for cross-validation ---
# This block assumes 'cleaned_df' and 'loso_candidate_files' are already defined
# and correctly populated before this script runs.

# The 'model' variable here is just a placeholder for the initial instantiation.
# It will be re-initialized inside the loop.
folds = get_loso_folds(
    cleaned_df,
    loso_candidate_files,
    interictal_val_ratio=0.2, # This ratio is for the global interictal split within folds
    random_state=42
)

all_metrics = []

# Define model_params once here, as it will be used to re-initialize the model in each fold
model_params = {
    'num_eeg_channels': 22,
    'num_classes': 2,
    'image_size': 224, # IMPORTANT: Input size for EEGHybridViT is 224x224
    # The EEGHybridViT class does not take patch_size, embed_dim, depth, heads, mlp_dim, dropout, classifier_dropout
    # in its __init__ (these are internal to the pretrained ViT).
    # Removed these params from model_params for EEGHybridViT.
}

for fold_idx, (train_df, val_df, held_out_filename_preictal) in enumerate(folds):
    print(f"\n==============================================")
    print(f"Starting Training for Fold {fold_idx+1}/{len(folds)}")
    print(f"Held-out PREICTAL file for Validation/Testing: {held_out_filename_preictal}")
    print(f"==============================================\n")
    
    # CRITICAL FIX: Re-initialize the model for each fold
    # Using EEGHybridViT here
    current_model_instance = EEGHybridViT(
        num_eeg_channels=model_params['num_eeg_channels'],
        num_classes=model_params['num_classes']
    )

    sanitized_filename = held_out_filename_preictal.replace('.npy', '').replace(':', '_').replace('.edf_stft_112x112', '')
    
    fold_model_path = os.path.join(
        "/kaggle/working/patient_models",
        f"best_model_patient_fold{fold_idx+1}_{sanitized_filename}.pt"
    )
    
    metrics = train_patient_model(
        train_df=train_df,
        val_df=val_df,
        model=current_model_instance,
        batch_size=8, # Use batch_size 8 from the high-accuracy code
        num_epochs=100,
        patience=20, # Use patience 20 from the high-accuracy code
        lr=1e-4, # Use lr 1e-4 from the high-accuracy code
        undersample=True,
        undersample_ratio=1.0, # Consolidate undersample_ratio to 1.0 for 1:1 balance
        accumulation_steps=1, # No explicit accumulation needed for batch_size 8 with this model
        model_path=fold_model_path
    )
    all_metrics.append(metrics)

# After the loop, `all_metrics` will contain a list of dictionaries, one for each fold.
df_metrics = pd.DataFrame(all_metrics)
print("\n--- Aggregated Metrics Across Folds ---")
print(df_metrics.mean(numeric_only=True))
print(df_metrics.head()) # Also show head of the final metrics DataFrame

df_metrics.to_csv('/kaggle/working/all_patient_fold_metrics.csv', index=False)
