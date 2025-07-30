import os
import numpy as np
import mne
from scipy.signal import stft

# --- PARAMETERS ---
l_freq = 1.0
h_freq = 60.0
segment_length_sec = 30
sfreq = 256  # Must match your data's sample rate
nperseg = 128
noverlap = 64
nfft = 512

standard_channels = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9',
    'FT9-FT10', 'FT10-T8'
]

def process_file_concat_stft(edf_path, patient_name, processed_dir):
    # Load and filter EEG
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks='eeg', fir_design='firwin', verbose=False)

    # Harmonize T8-P8 channel
    extra_ch = {}
    if 'T8-P8-0' in raw.ch_names and 'T8-P8-1' in raw.ch_names:
        t8p8_0 = raw.get_data(picks=['T8-P8-0'])[0]
        t8p8_1 = raw.get_data(picks=['T8-P8-1'])[0]
        t8p8_avg = (t8p8_0 + t8p8_1) / 2
        raw.drop_channels(['T8-P8-0', 'T8-P8-1'])
        extra_ch['T8-P8'] = t8p8_avg
    elif 'T8-P8-0' in raw.ch_names:
        raw.rename_channels({'T8-P8-0': 'T8-P8'})
    elif 'T8-P8-1' in raw.ch_names:
        raw.rename_channels({'T8-P8-1': 'T8-P8'})

    # Build data array in canonical order
    data = []
    for ch in standard_channels:
        if ch in raw.ch_names:
            data.append(raw.get_data(picks=[ch])[0])
        elif ch in extra_ch:
            data.append(extra_ch[ch])
        else:
            data.append(np.zeros(raw.n_times))
    data = np.array(data)

    # Standardized MNE object, average reference
    info = mne.create_info(standard_channels, raw.info['sfreq'], ch_types='eeg')
    raw_standard = mne.io.RawArray(data, info, verbose=False)
    raw_standard.set_eeg_reference(ref_channels='average', verbose=False)

    # Cache data for efficient access
    segment_data = raw_standard.get_data()
    segment_length_samples = int(segment_length_sec * sfreq)
    n_channels, n_samples = segment_data.shape
    num_segments = n_samples // segment_length_samples

    # Segmentation and per-segment normalization
    segments_norm = []
    for i in range(num_segments):
        segment = segment_data[:, i*segment_length_samples:(i+1)*segment_length_samples]
        segment_norm = (segment - segment.mean(axis=1, keepdims=True)) / (segment.std(axis=1, keepdims=True) + 1e-8)
        segments_norm.append(segment_norm)
    segments_norm = np.array(segments_norm, dtype=np.float32)  # shape: (segments, channels, segment_len)

    # Save
    patient_save_dir = os.path.join(processed_dir, patient_name)
    os.makedirs(patient_save_dir, exist_ok=True)
    base_name = os.path.basename(edf_path).replace('.edf', '_segmented.npy')
    save_path = os.path.join(patient_save_dir, base_name)
    np.save(save_path, segments_norm)
    print(f"saved_file {save_path}")
    return save_path

# --- Batch Processing Example ---
base_dir = '/content/drive/MyDrive/Seizure_Dataset/Original_dataset'
processed_dir = '/content/drive/MyDrive/Seizure_Dataset/Processed_Segments_30sec'

for patient_name in os.listdir(base_dir):
    patient_dir = os.path.join(base_dir, patient_name)
    if not os.path.isdir(patient_dir):
        continue
    for file_name in os.listdir(patient_dir):
        if not file_name.endswith('.edf'):
            continue
        edf_path = os.path.join(patient_dir, file_name)
        print(f"Processing file {edf_path}")
        process_file_concat_stft(edf_path, patient_name, processed_dir)

print("All spectrogram segments have been saved.")
