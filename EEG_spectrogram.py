import os
import numpy as np
from scipy.signal import stft
from PIL import Image

# --- Parameters ---
sfreq = 256
nperseg = 256
noverlap = 128
nfft = 512

# Patients to process
patients_to_use = [
    'Patient_1', 'Patient_3', 'Patient_5', 'Patient_7',
    'Patient_11', 'Patient_14', 'Patient_16', 'Patient_19'
]

processed_dir = '/content/drive/MyDrive/Seizure_Dataset/Processed_Segments_30sec/'
out_dir = '/content/drive/MyDrive/Seizure_Dataset/STFT_30sec/'
os.makedirs(out_dir, exist_ok=True)

def compute_and_save_stft(npy_path, patient_name, save_dir):
    segments = np.load(npy_path)  # (segments, channels, samples)
    if segments.ndim != 3:
        raise ValueError(f"Expected (segments, channels, samples), got {segments.shape}")

    n_segments, n_channels, _ = segments.shape

    stft_spectrograms = []
    for i in range(n_segments):
        seg = segments[i]  # (channels, samples)
        seg_stft = []
        for ch in range(n_channels):
            ch_data = seg[ch]
            seg_len = ch_data.shape[0]
            nperseg_eff = min(nperseg, seg_len)
            noverlap_eff = min(noverlap, nperseg_eff - 1) if nperseg_eff > 1 else 0
            _, _, Zxx = stft(
                ch_data,
                fs=sfreq,
                window='hann',
                nperseg=nperseg_eff,
                noverlap=noverlap_eff,
                nfft=nfft,
                boundary=None
            )
            mag = np.abs(Zxx)
            mag_img = Image.fromarray(mag)
            seg_stft.append(np.array(mag_img, dtype=np.float32))
        seg_stft = np.stack(seg_stft, axis=0)  # (channels, 112, 112)
        stft_spectrograms.append(seg_stft)
    stft_spectrograms = np.array(stft_spectrograms, dtype=np.float16)  # (segments, channels, 112, 112)

    # Save output
    save_patient_dir = os.path.join(save_dir, patient_name)
    os.makedirs(save_patient_dir, exist_ok=True)
    base_name = os.path.basename(npy_path).replace('_segmented.npy', '_stft_30sec.npy')
    save_path = os.path.join(save_patient_dir, base_name)
    np.save(save_path, stft_spectrograms)
    print(f"Saved STFT spectrogram for {patient_name}: {save_path}")

# Batch processing for selected patients
for patient_name in patients_to_use:
    patient_dir = os.path.join(processed_dir, patient_name)
    if not os.path.isdir(patient_dir):
        print(f"Patient directory not found: {patient_name}")
        continue
    for file_name in os.listdir(patient_dir):
        if not file_name.endswith('_segmented.npy'):
            continue
        npy_path = os.path.join(patient_dir, file_name)
        print(f"Processing {patient_name}, file: {file_name}")
        compute_and_save_stft(npy_path, patient_name, out_dir)

print("All 112x112 STFT spectrograms generated and saved.")

import os
import numpy as np
import pandas as pd

# ---- CONFIGURATION ----
SEGMENT_DURATION = 30  # seconds
SUMMARY_FILE = '/content/drive/MyDrive/Seizure_Dataset/Summary/chb01-summary.txt'
SEGMENTED_DIR = '/content/drive/MyDrive/Seizure_Dataset/Processed_Segments_30sec/Patient_1'
OUTPUT_DF = '/content/drive/MyDrive/Seizure_Dataset/Summary_30sec/chb1_segments.csv'
PATIENT_NAME = 'chb01'

def parse_time(timestr):
    h, m, s = map(int, timestr.strip().split(":"))
    return h * 3600 + m * 60 + s

def extract_time_from_line(line):
    return line.split(":", 1)[1].strip()

def parse_summary(summary_file):
    file_metadata = []
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("File Name:"):
            fname = line.split(":", 1)[1].strip()
            start_time = parse_time(extract_time_from_line(lines[i+1]))
            end_time = parse_time(extract_time_from_line(lines[i+2]))
            n_seizures = int(lines[i+3].split(":", 1)[1].strip())
            seizures = []
            for s in range(n_seizures):
                sz_start = int(lines[i+4+s*2].split(":", 1)[1].strip().split()[0])
                sz_end = int(lines[i+5+s*2].split(":", 1)[1].strip().split()[0])
                seizures.append((sz_start, sz_end))
            file_metadata.append({
                'filename': fname,
                'start_time': start_time,
                'end_time': end_time,
                'seizures': seizures
            })
            i += 4 + n_seizures * 2
        else:
            i += 1
    return file_metadata

def label_segments_filewise(meta, seg_file, seg_path, segment_duration=5):
    segments = np.load(seg_path)
    n_segments = segments.shape[0]
    file_labels = []
    seizures = meta['seizures']

    if len(seizures) == 0:
        # No seizure in file: all interictal
        for i in range(n_segments):
            seg_start = i * segment_duration
            seg_end = seg_start + segment_duration
            file_labels.append((i, seg_start, seg_end, 0, 'interictal'))
        return file_labels

    # For each segment, check against all seizures in the file
    for i in range(n_segments):
        seg_start = i * segment_duration
        seg_end = seg_start + segment_duration

        label = 4  # Default: grey
        seg_type = 'grey'

        for sz_start, sz_end in seizures:
            # Ictal: segment overlaps with any seizure
            if seg_start < sz_end and seg_end > sz_start:
                label = 2
                seg_type = 'ictal'
                break  # Highest priority

        if label != 2:
            for sz_start, sz_end in seizures:
                # Preictal: 0–60 min before any seizure onset
                if sz_start - 60*60 <= seg_start < sz_start:
                    label = 1
                    seg_type = 'preictal'
                    break  # Preictal priority after ictal

        if label == 4:
            # Interictal: ≥4h away from all seizures
            interictal = True
            for sz_start, sz_end in seizures:
                if not (seg_end <= sz_start - 4*3600 or seg_start >= sz_end + 4*3600):
                    interictal = False
                    break
            if interictal:
                label = 0
                seg_type = 'interictal'

        file_labels.append((i, seg_start, seg_end, label, seg_type))
    return file_labels

def main():
    file_metadata = parse_summary(SUMMARY_FILE)
    all_segments = []

    for meta in file_metadata:
        edf_base = meta['filename'].replace('.edf', '')
        seg_file = f"{edf_base}_segmented.npy"
        seg_path = os.path.join(SEGMENTED_DIR, seg_file)
        if not os.path.exists(seg_path):
            print(f"Missing: {seg_path}")
            continue

        file_labels = label_segments_filewise(meta, seg_file, seg_path, segment_duration=SEGMENT_DURATION)
        for (i, seg_start, seg_end, label, seg_type) in file_labels:
            all_segments.append({
                'segment': i,
                'filename': seg_file,
                'patient': PATIENT_NAME,
                'segment_start': seg_start,
                'segment_end': seg_end,
                'segment_type': seg_type,
                'segment_label': label
            })

    df = pd.DataFrame(all_segments)
    df.to_csv(OUTPUT_DF, index=False)
    print(f"Saved DataFrame to {OUTPUT_DF} with {len(df)} segments.")

if __name__ == "__main__":
    main()
