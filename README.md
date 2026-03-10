# LightPollutionPonds

# Light-pollution-and-pond-soundscapes

## Power spectral density analysis
```
import os
import shutil
import datetime
import re

# ================= CONFIGURATION =================

# 1. ENTER YOUR PATHS HERE
SOURCE_DIRECTORY = r"D:\Light pollution\Python"
OUTPUT_DIRECTORY = r"D:\Light pollution\Sorted folders"

# 2. DEFINING THE PHASES
# I have corrected the "End Times" based on your confirmation 
# that every phase is exactly 1 hour long.

FOLDER_CONFIG = {
    "OSP_2_25.08.2025": {
        "base_date_str": "2025-08-25",
        "phases": [
            {"name": "Phase_I",   "start": "21:11", "end": "22:10"},
            {"name": "Phase_II",  "start": "22:11", "end": "23:10"},
            {"name": "Phase_III", "start": "23:11", "end": "00:10"} 
        ]
    },
    "OSP_3_27.08.2025": {
        "base_date_str": "2025-08-27",
        "phases": [
            {"name": "Phase_I",   "start": "21:07", "end": "22:06"},
            {"name": "Phase_II",  "start": "22:07", "end": "23:06"},
            {"name": "Phase_III", "start": "23:07", "end": "00:06"}
        ]
    },
    "OSP_1_26_08_24": {
        "base_date_str": "2024-08-26",
        "phases": [
            {"name": "Phase_I",   "start": "21:08", "end": "22:07"},
            {"name": "Phase_II",  "start": "22:08", "end": "23:07"},
            {"name": "Phase_III", "start": "23:08", "end": "00:07"}
        ]
    },
    "UoBBG_29.08.2024": {
        "base_date_str": "2024-08-28", 
        "phases": [
            {"name": "Phase_I",   "start": "21:04", "end": "22:03"},
            {"name": "Phase_II",  "start": "22:04", "end": "23:03"},
            {"name": "Phase_III", "start": "23:04", "end": "00:03"}
        ]
    },
    "FF_04.09.2024": {
        "base_date_str": "2024-09-04", 
        "phases": [
            {"name": "Phase_I",   "start": "20:49", "end": "21:48"},
            {"name": "Phase_II",  "start": "21:49", "end": "22:48"},
            {"name": "Phase_III", "start": "22:49", "end": "23:48"}
        ]
    },
    "EF_03.09.2024": {
        "base_date_str": "2024-09-03",
        "phases": [
            {"name": "Phase_I",   "start": "20:51", "end": "21:50"},
            {"name": "Phase_II",  "start": "21:51", "end": "22:50"},
            {"name": "Phase_III", "start": "22:51", "end": "23:50"}
        ]
    }
}

# ================= LOGIC =================

def get_datetime_from_filename(filename):
    """
    Extracts datetime from filename string like '20240903_231700'.
    Returns a datetime object or None if format not found.
    """
    # Regex looks for 8 digits (date) followed by _ followed by 6 digits (time)
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        date_part = match.group(1)
        time_part = match.group(2)
        try:
            return datetime.datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    return None

def process_audio_files():
    print("Starting processing using FILENAMES...")
    
    for folder_name, config in FOLDER_CONFIG.items():
        folder_path = os.path.join(SOURCE_DIRECTORY, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Skipping: {folder_name} (Folder not found)")
            continue

        print(f"Processing Folder: {folder_name}")
        
        # Base date for this experiment folder
        base_date = datetime.datetime.strptime(config['base_date_str'], "%Y-%m-%d")
        
        # 1. Collect all valid wav files and parse their times
        valid_files = []
        for f in os.listdir(folder_path):
            if f.lower().endswith('.wav'):
                file_dt = get_datetime_from_filename(f)
                if file_dt:
                    full_path = os.path.join(folder_path, f)
                    valid_files.append((full_path, file_dt, f))
                else:
                    print(f"  [Warning] Could not parse date from filename: {f}")
        
        # Sort files by their actual timestamp
        valid_files.sort(key=lambda x: x[1])

        # 2. Iterate through phases
        for phase in config['phases']:
            # Create Time Objects
            t_start = datetime.datetime.strptime(phase['start'], "%H:%M").time()
            t_end = datetime.datetime.strptime(phase['end'], "%H:%M").time()
            
            # Combine with Date to make full Datetime objects
            # Start time is always on the base date
            dt_start = datetime.datetime.combine(base_date, t_start)
            
            # End time: if end time is smaller than start time (e.g. 23:00 to 00:00), 
            # it implies it rolled over to the next day.
            dt_end = datetime.datetime.combine(base_date, t_end)
            if t_end < t_start:
                dt_end += datetime.timedelta(days=1)
            
            # 3. Find files that match this window
            phase_files = []
            for f_path, f_dt, f_name in valid_files:
                if dt_start <= f_dt <= dt_end:
                    phase_files.append((f_path, f_name))
            
            # 4. Copy files (Limit to 60 if needed, usually safer to take all in window)
            # You requested 60 files chunks, so we slice [:60]
            files_to_copy = phase_files[:60]
            
            print(f"  > {phase['name']} ({dt_start} - {dt_end})")
            print(f"    Found {len(phase_files)} files. Copying {len(files_to_copy)}...")

            if files_to_copy:
                dest_dir = os.path.join(OUTPUT_DIRECTORY, folder_name, phase['name'])
                os.makedirs(dest_dir, exist_ok=True)
                
                for src, fname in files_to_copy:
                    dst = os.path.join(dest_dir, fname)
                    shutil.copy2(src, dst)
            else:
                print(f"    No files found for this phase.")

    print("\nProcessing complete.")

if __name__ == "__main__":
    process_audio_files()

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

# ================= CONFIGURATION =================
# Point this to your main "Sorted folders" directory
ROOT_DIRECTORY = r"D:\Light pollution\Sorted folders"

# FFT Parameters for 93.75Hz resolution at 96kHz
FS = 96000 
NPERSEG = 1024 

def run_spectral_analysis():
    print(f"Starting Analysis (Sample Rate: {FS}Hz)...")

    # Walk through the directory structure
    for root, dirs, files in os.walk(ROOT_DIRECTORY):
        # Filter for wav files
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        
        if not wav_files:
            continue

        # Identifying folder and phase names for the output file
        # Assumes structure: ...\Sorted folders\Experiment_Folder\Phase_X
        path_parts = root.split(os.sep)
        phase_name = path_parts[-1]
        experiment_name = path_parts[-2]
        
        print(f"Processing: {experiment_name} -> {phase_name}")

        all_psds = []

        for wav_name in wav_files:
            file_path = os.path.join(root, wav_name)
            try:
                sample_rate, data = wavfile.read(file_path)
                
                # Check if sample rate matches
                if sample_rate != FS:
                    # Optional: print warning if a file differs from 96kHz
                    pass

                # Handle Stereo (use left channel)
                if len(data.shape) > 1:
                    data = data[:, 0]

                # Convert to float and normalize to -1.0 to 1.0
                # This ensures dB levels are relative to Digital Full Scale (dBFS)
                data = data.astype(float) / 32768.0 if data.dtype == np.int16 else data.astype(float)

                # Calculate Power Spectral Density (PSD)
                # nperseg=1024 creates the 93.75Hz bins
                freqs, psd = welch(data, fs=FS, nperseg=NPERSEG, scaling='spectrum')
                all_psds.append(psd)

            except Exception as e:
                print(f"  [Error] Could not read {wav_name}: {e}")

        if all_psds:
            # Average all minutes in the phase
            avg_psd = np.mean(all_psds, axis=0)
            
            # Convert power to dB Level
            # Formula: 10 * log10(Power) is equivalent to 20 * log10(Amplitude)
            levels_db = 10 * np.log10(avg_psd + 1e-12)

            # Generate Output Filename
            out_name = f"spectrum_{experiment_name}_{phase_name}.txt"
            output_path = os.path.join(root, out_name)

            # Save in the requested Tab-Separated format
            with open(output_path, 'w') as f:
                f.write("Frequency (Hz)\tLevel (dB)\n")
                # We start from index 1 to skip the 0Hz (DC) component
                for i in range(1, len(freqs)):
                    f.write(f"{freqs[i]:.6f}\t{levels_db[i]:.6f}\n")
            
            print(f"  Saved: {out_name}")

    print("\nProcessing complete. Check your phase folders for the .txt files.")

if __name__ == "__main__":
    run_spectral_analysis()
```

## Acoustic feature extraction and signal processing

```
#By Jack A. Greenhalgh, June, 2025.
#Department of Biology, McGill University, 1205 Dr Penfield Ave, Montreal, Quebec, H3A 1B1, Canada.
import os
import numpy as np
import pandas as pd
import librosa
import gc
from scipy.signal import find_peaks, butter, lfilter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor as Executor, as_completed

# ==============================================================================
# 1. USER CONFIGURATION (CHANGE THESE ONLY)
# ==============================================================================
# Define the frequency range you want to analyze (in Hz)
F_MIN = 40000    # Lower limit of the bandpass filter
F_MAX = 47000   # Upper limit of the bandpass filter

# Define the "split point" between Human noise (Anthro) and Nature (Bio)
NDSI_SPLIT = 42800 

# Directory settings
main_directory = r"D:\Light pollution\Python"

# ==============================================================================
# 2. AUTOMATIC CALCULATION
# ==============================================================================
# Automatically set Sampling Rate (SR) to 2.5x the max frequency
TARGET_SR = int(F_MAX * 2.5)

print(f"--- AUTO-CONFIGURATION ---")
print(f"Analysis Band: {F_MIN} Hz to {F_MAX} Hz")
print(f"NDSI Split:    Anthro ({F_MIN}-{NDSI_SPLIT} Hz) | Bio ({NDSI_SPLIT}-{F_MAX} Hz)")
print(f"Processing SR: {TARGET_SR} Hz (Automatically calculated)")
print("-" * 26)

# Map variables to the processing logic
low_cut = float(F_MIN)
high_cut = float(F_MAX)
anthro_min, anthro_max = F_MIN, NDSI_SPLIT
bio_min, bio_max = NDSI_SPLIT, F_MAX

# Standard fixed parameters
peak_height_db = -80  
min_peak_dist = int(TARGET_SR / 10) # Dynamic: roughly 0.1 seconds
n_mfcc = 13  
n_fft = 1024 
hop_length = 256

# =============================
# Signal Processing Functions
# =============================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Safety check: if high band is too close to Nyquist, cap it at 99%
    if high >= 0.99: 
        high = 0.99 
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def calculate_ndsi(S, frequencies, anthro_range, bio_range):
    # Mask: Select rows in the spectrogram corresponding to the frequency ranges
    anthro_mask = (frequencies >= anthro_range[0]) & (frequencies <= anthro_range[1])
    bio_mask = (frequencies >= bio_range[0]) & (frequencies <= bio_range[1])
    
    anthro_energy = np.sum(S[anthro_mask, :])
    bio_energy = np.sum(S[bio_mask, :])
    
    total_energy = anthro_energy + bio_energy
    
    if total_energy > 0:
        ndsi = (bio_energy - anthro_energy) / total_energy
    else:
        ndsi = 0
        
    return ndsi, bio_energy, anthro_energy

def process_file(foldername, folder_path, filename):
    filepath = os.path.join(folder_path, filename)
    features = {'filename': f"{foldername}_{filename}"}

    try:
        # 1. Load & Downsample using the AUTO-CALCULATED SR
        y, sr = librosa.load(filepath, sr=TARGET_SR)
        
        if len(y) == 0: return None

        # 2. Apply Bandpass Filter
        y_filtered = butter_bandpass_filter(y, low_cut, high_cut, sr)

        # 3. Frequency Domain Analysis
        S_mag = np.abs(librosa.stft(y_filtered, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Features
        features['Spectral_Centroid_Mean'] = np.mean(librosa.feature.spectral_centroid(S=S_mag, sr=sr))
        features['Spectral_Flatness_Mean'] = np.mean(librosa.feature.spectral_flatness(S=S_mag))
        
        # NDSI using auto-ranges
        ndsi, bio_en, anthro_en = calculate_ndsi(S_mag, freqs, (anthro_min, anthro_max), (bio_min, bio_max))
        features['NDSI'] = ndsi
        features['Bio_Energy'] = bio_en
        features['Anthro_Energy'] = anthro_en

        # 4. Temporal Analysis
        features['ZCR_Mean'] = np.mean(librosa.feature.zero_crossing_rate(y_filtered))
        features['RMS_Mean'] = np.mean(librosa.feature.rms(y=y_filtered))
        
        # Event Detection
        peaks, _ = find_peaks(np.abs(y_filtered), height=librosa.db_to_amplitude(peak_height_db), distance=min_peak_dist)
        features['Event_Count'] = len(peaks)

        # 5. MFCCs
        mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(S_mag + 1e-10), sr=sr, n_mfcc=n_mfcc)
        for i, coeff in enumerate(np.mean(mfccs, axis=1)):
            features[f'MFCC_{i+1}'] = coeff

        # Cleanup
        del y, y_filtered, S_mag, freqs
        gc.collect() 
        return features

    except Exception as e:
        print(f"⚠ Error in {filename}: {e}")
        return None

# =============================
# Main Execution Logic
# =============================
def main():
    foldernames = [f for f in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, f))]
    
    for foldername in foldernames:
        folder_path = os.path.join(main_directory, foldername)
        wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
        if not wav_files: continue

        print(f"\n🚀 Analyzing {foldername} ({len(wav_files)} files)")
        results = []

        with Executor(max_workers=2) as executor:
            futures = {executor.submit(process_file, foldername, folder_path, f): f for f in wav_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting {int(F_MIN)}-{int(F_MAX)}Hz"):
                res = future.result()
                if res: results.append(res)

        if results:
            # FIX: Ensure we use 'df' not 'dr'
            df = pd.DataFrame(results)
            
            # FIX: Correctly format the range label
            range_label = f"{int(F_MIN)}-{int(F_MAX)}Hz"
            
            # Construct output path
            out_name = os.path.join(folder_path, f"{foldername}_Summary_{range_label}")
            
            # Save files
            df.to_csv(f"{out_name}.csv", index=False, encoding='utf-8-sig')
            df.to_csv(f"{out_name}.txt", index=False, sep=',', encoding='utf-8')
            print(f"✨ Success: Saved to {out_name}.csv")

if __name__ == "__main__":
    main()
```

## Spectral power analysis: visualisation 

```
library(ggplot2)
library(dplyr)
library(stringr)
library(purrr)

# Directory containing spectra
data_dir <- "C:/Users/Administrador/OneDrive - McGill University/Light pollution and pond soundscapes/Jan 2026/.txt files for Bode magnitude plot"

# List all .txt files
files <- list.files(
  path = data_dir,
  pattern = "\\.txt$",
  full.names = TRUE
)

# Read and annotate all files
spectra_all <- map_dfr(files, function(f) {
  
  df <- read.table(f, header = TRUE, sep = "\t")
  fname <- basename(f)
  
  phase <- str_extract(fname, "Phase_[I]{1,3}")
  
  site <- fname |>
    str_remove("^spectrum_") |>
    str_remove("_[0-9]{2}.*") |>
    str_remove("_Phase_[I]{1,3}")
  
  df %>%
    mutate(
      Site  = site,
      Phase = phase
    )
})

# Order phases
spectra_all$Phase <- factor(
  spectra_all$Phase,
  levels = c("Phase_I", "Phase_II", "Phase_III")
)

# Phase colors
phase_colors <- c(
  "Phase_I"   = "black",
  "Phase_II"  = "#E69F00",
  "Phase_III" = "#56B4E9"
)

# Output directory
out_dir <- file.path(data_dir, "Bode_plots")
dir.create(out_dir, showWarnings = FALSE)

# ===============================
# GLOBAL AXIS LIMITS
# ===============================

fmax <- 46000  # 46 kHz

ymin <- floor(min(spectra_all$Level..dB., na.rm = TRUE) / 2) * 2
ymax <- ceiling(max(spectra_all$Level..dB., na.rm = TRUE) / 2) * 2

y_breaks <- seq(ymin, ymax, by = 2)

# ===============================
# Plot one Bode magnitude per site
# ===============================

sites <- unique(spectra_all$Site)

for (s in sites) {
  
  df_site <- spectra_all %>%
    filter(Site == s, Frequency..Hz. <= fmax)
  
  p <- ggplot(
    df_site,
    aes(x = Frequency..Hz., y = Level..dB., color = Phase)
  ) +
    geom_line() +
    scale_color_manual(values = phase_colors) +
    scale_x_continuous(
      name = "Frequency (kHz)",
      limits = c(0, fmax),
      breaks = seq(0, fmax, by = 5000),
      labels = function(x) x / 1000
    ) +
    scale_y_continuous(
      name = "Magnitude (dB)",
      limits = c(ymin, ymax),
      breaks = y_breaks
    ) +
    labs(
      title = paste("Bode Magnitude Spectrum –", s),
      color = "Phase"
    ) +
    theme_bw()
  
  ggsave(
    filename = file.path(out_dir, paste0("Bode_", s, "_46_kHz.jpeg")),
    plot = p,
    dpi = 300,
    width = 8,
    height = 6,
    units = "in"
  )
}

```

## Dimensionality reduction and statistical modeling


```
# =====================================================
# 0. LIBRARIES & PARALLEL SETUP
# =====================================================
library(dplyr)
library(stringr)
library(lubridate)
library(tidyr)
library(ggplot2)
library(nlme)
library(foreach)
library(doSNOW) 
library(caret)
library(MuMIn) # For R-squared
library(car)   # For diagnostics

set.seed(42)

# Set up the cluster
n_cores <- parallel::detectCores() - 1
cl <- makeCluster(n_cores)
registerDoSNOW(cl)

# =====================================================
# 1. IMPORT & INITIAL PROCESSING
# =====================================================
dir_path <- "C:/Users/Administrador/OneDrive - McGill University/Light pollution and pond soundscapes/Feb 2026/Without NDSI & Anthro_Energy/All data"
setwd(dir_path)

txt_files <- list.files(path = ".", pattern = "\\.txt$", full.names = TRUE)

df_global_raw <- lapply(txt_files, function(f) {
  dat <- read.csv(f, stringsAsFactors = FALSE)
  dat$source_txt <- basename(f) 
  return(dat)
}) %>% bind_rows()

# =====================================================
# 2. METADATA & TREATMENT ASSIGNMENT
# =====================================================
df_global_proc <- df_global_raw %>%
  mutate(
    Site = case_when(
      str_detect(filename, "EF")    ~ "EF",
      str_detect(filename, "FWF|FF") ~ "FF",
      str_detect(filename, "OSP")   ~ "OSP",
      str_detect(filename, "UoBBG") ~ "UoBBG",
      TRUE                          ~ NA_character_
    ),
    txt_low = tolower(source_txt),
    Bandwidth = case_when(
      str_detect(txt_low, "1-10khz")  ~ "1 - 10 kHz",
      str_detect(txt_low, "10-20khz") ~ "10 - 20 kHz",
      str_detect(txt_low, "20-30khz") ~ "20 - 30 kHz",
      str_detect(txt_low, "30-40khz") ~ "30 - 40 kHz",
      str_detect(txt_low, "40-47khz") ~ "40 - 47 kHz",
      str_detect(txt_low, "7-14khz")  ~ "7 - 14 kHz",
      str_detect(txt_low, "2-5khz")   ~ "2 - 5 kHz",
      TRUE                            ~ "Other"
    ),
    raw_ts = str_extract(filename, "\\d{8}_\\d{6}"),
    Datetime = as.POSIXct(raw_ts, format = "%Y%m%d_%H%M%S", tz = "Europe/London"),
    Exp_Date = as.Date(Datetime)
  ) %>%
  filter(!is.na(Site), !is.na(Datetime), Bandwidth != "Other")

assign_treatment <- function(Site, Exp_Date) {
  if (Site == "OSP" && Exp_Date == as.Date("2025-08-25")) {
    data.frame(Treatment = c("Phase I", "Phase II", "Phase III"),
               S = c("21:11", "22:11", "23:11"), E = c("22:10", "23:10", "00:10"))
  } else if (Site == "OSP" && Exp_Date == as.Date("2025-08-27")) {
    data.frame(Treatment = c("Phase I", "Phase II", "Phase III"),
               S = c("21:07", "22:07", "23:07"), E = c("22:06", "23:06", "00:06"))
  } else if (Site == "OSP") {
    data.frame(Treatment = c("Phase I", "Phase II", "Phase III"),
               S = c("21:08", "22:08", "23:08"), E = c("22:07", "23:07", "00:07"))
  } else if (Site == "UoBBG") {
    data.frame(Treatment = c("Phase I", "Phase II", "Phase III"),
               S = c("21:04", "22:04", "23:04"), E = c("22:03", "23:03", "00:03"))
  } else if (Site == "FF") {
    data.frame(Treatment = c("Phase I", "Phase II", "Phase III"),
               S = c("20:49", "21:49", "22:49"), E = c("21:48", "22:48", "23:48"))
  } else { 
    data.frame(Treatment = c("Phase I", "Phase II", "Phase III"),
               S = c("20:51", "21:51", "22:51"), E = c("21:50", "22:50", "23:50"))
  }
}

df_global_treated <- df_global_proc %>%
  rowwise() %>%
  mutate(phases = list(assign_treatment(Site, Exp_Date))) %>%
  ungroup() %>%
  tidyr::unnest(phases) %>%
  mutate(
    Start = as.POSIXct(paste(Exp_Date, S), format = "%Y-%m-%d %H:%M", tz = "Europe/London"),
    End = as.POSIXct(ifelse(E < S, paste(Exp_Date + 1, E), paste(Exp_Date, E)),
                     format = "%Y-%m-%d %H:%M", tz = "Europe/London")
  ) %>%
  filter(Datetime >= Start & Datetime <= End) %>%
  mutate(
    Treatment = factor(Treatment, levels = c("Phase I", "Phase II", "Phase III")),
    Bandwidth = as.factor(Bandwidth)
  )

# =====================================================
# 3. GLOBAL PCA
# =====================================================
priority_vars <- c("ACI", "Bio_Energy", "Event_Count", "ZCR_Mean")
spectral_vars <- c("ADI", "RMS_Mean", paste0("MFCC_", 1:13))
pca_vars_full <- c(priority_vars, spectral_vars)

df_scaled <- df_global_treated %>%
  mutate(across(all_of(pca_vars_full), ~ as.numeric(scale(.))))

cor_matrix <- cor(df_scaled[, pca_vars_full], use="complete.obs")
to_remove_indices <- findCorrelation(cor_matrix, cutoff=0.8)
removed_names <- pca_vars_full[to_remove_indices]
pca_vars <- setdiff(pca_vars_full, setdiff(removed_names, priority_vars))

pca_res <- prcomp(df_scaled[, pca_vars], center=FALSE, scale.=FALSE)

# Combine PCA results with metadata
df_model_raw <- cbind(df_scaled, PC1 = pca_res$x[,1], PC2 = pca_res$x[,2]) %>%
  arrange(Site, Exp_Date, Bandwidth, Datetime)

# =====================================================
# 4. SENSITIVITY FILTERING (NEW GATEKEEPER)
# =====================================================
cat("\nCALCULATING PCA SENSITIVITY PER BAND...\n")

band_variance_list <- list()
for(b in unique(df_model_raw$Bandwidth)) {
  band_sub <- df_model_raw %>% filter(Bandwidth == b)
  original_vars_band <- band_sub[, pca_vars]
  
  # Total local variance in this specific band
  total_var_band <- sum(apply(original_vars_band, 2, var, na.rm = TRUE))
  
  # Variance of this band captured by the Global PC1 and PC2
  pc_vars_band <- sum(var(band_sub$PC1, na.rm = TRUE), 
                      var(band_sub$PC2, na.rm = TRUE))
  
  band_variance_list[[b]] <- data.frame(
    Bandwidth = b,
    Percent_Explained = (pc_vars_band / total_var_band) * 100
  )
}

Band_PCA_Sensitivity <- bind_rows(band_variance_list)

# Identify bands that meet the 10% threshold
sensitive_bands <- Band_PCA_Sensitivity %>% 
  filter(Percent_Explained >= 10) %>% 
  pull(Bandwidth)

# Final Filtering of the model data
df_model <- df_model_raw %>% filter(Bandwidth %in% sensitive_bands)

cat("Bands Kept (>=10% var):", paste(sensitive_bands, collapse=", "), "\n")
cat("Bands Removed (<10% var):", paste(setdiff(unique(df_model_raw$Bandwidth), sensitive_bands), collapse=", "), "\n")

# =====================================================
# 5. MAIN LOOP: BLOCK PERMUTATIONS
# =====================================================
bands <- unique(df_model$Bandwidth)
pcs_to_model <- c("PC1", "PC2")
n_perms <- 999
all_results_list <- list()

total_iterations <- length(pcs_to_model) * length(bands)
pb <- txtProgressBar(max = total_iterations, style = 3)
counter <- 0

cat("\nSTARTING BLOCK PERMUTATION MODELS ON SENSITIVE BANDS\n")

for(current_pc in pcs_to_model) {
  for(current_band in bands) {
    counter <- counter + 1
    setTxtProgressBar(pb, counter)
    
    sub_data <- df_model %>% filter(Bandwidth == current_band)
    model_formula <- as.formula(paste(current_pc, "~ Treatment"))
    
    obs_model <- try(lme(
      fixed = model_formula,
      random = ~ 1 | Site/Exp_Date, 
      correlation = corAR1(form = ~ 1 | Site/Exp_Date),
      data = sub_data,
      control = lmeControl(opt = "optim", msMaxIter = 200)
    ), silent = TRUE)
    
    if(inherits(obs_model, "try-error")) next
    
    summ_tab <- summary(obs_model)$tTable
    obs_t_p2 <- summ_tab["TreatmentPhase II", "t-value"]
    obs_t_p3 <- summ_tab["TreatmentPhase III", "t-value"]
    
    null_dist <- foreach(i = 1:n_perms, .combine = 'rbind', .packages = c('nlme', 'dplyr')) %dopar% {
      p_data <- sub_data %>%
        group_by(Site) %>%
        mutate(Treatment = sample(Treatment)) %>% 
        ungroup() %>%
        mutate(Treatment = factor(Treatment, levels = c("Phase I", "Phase II", "Phase III")))
      
      p_mod <- try(lme(fixed = model_formula, 
                       random = ~ 1 | Site/Exp_Date, 
                       correlation = corAR1(form = ~ 1 | Site/Exp_Date),
                       data = p_data, 
                       control = lmeControl(opt = "optim", msMaxIter = 100)), silent = TRUE)
      
      if(!inherits(p_mod, "try-error")) {
        tt <- summary(p_mod)$tTable
        return(c(p2 = tt["TreatmentPhase II", "t-value"], p3 = tt["TreatmentPhase III", "t-value"]))
      } else { return(c(p2 = NA, p3 = NA)) }
    }
    
    null_dist <- as.data.frame(null_dist)
    valid_p2 <- na.omit(null_dist$p2); valid_p3 <- na.omit(null_dist$p3)
    
    p_val_p2 <- (sum(abs(valid_p2) >= abs(obs_t_p2)) + 1) / (length(valid_p2) + 1)
    p_val_p3 <- (sum(abs(valid_p3) >= abs(obs_t_p3)) + 1) / (length(valid_p3) + 1)
    
    all_results_list[[paste0(current_pc, "_", current_band)]] <- data.frame(
      Component = current_pc, Bandwidth = current_band,
      Beta_P2 = summ_tab["TreatmentPhase II", "Value"], SE_P2 = summ_tab["TreatmentPhase II", "Std.Error"], 
      Perm_P_P2 = p_val_p2, Count_P2 = length(valid_p2),
      Beta_P3 = summ_tab["TreatmentPhase III", "Value"], SE_P3 = summ_tab["TreatmentPhase III", "Std.Error"], 
      Perm_P_P3 = p_val_p3, Count_P3 = length(valid_p3)
    )
  }
}
close(pb)
stopCluster(cl)

# =====================================================
# 6. FINAL VISUALIZATION (FIXED LINES & SORTING)
# =====================================================
target_order <- c("2 - 5 kHz", "7 - 14 kHz", "1 - 10 kHz", "20 - 30 kHz")

Final_Results <- bind_rows(all_results_list) %>%
  mutate(Perm_P_P2_Adj = p.adjust(Perm_P_P2, method = "fdr"),
         Perm_P_P3_Adj = p.adjust(Perm_P_P3, method = "fdr"))

plot_data <- Final_Results %>%
  filter(Bandwidth %in% target_order) %>%
  pivot_longer(cols = -c(Component, Bandwidth), names_to = "raw_name", values_to = "val") %>%
  mutate(Phase = ifelse(str_detect(raw_name, "P2"), "Phase II (Light)", "Phase III (Recovery)"),
         Metric = case_when(str_detect(raw_name, "Beta") ~ "Beta",
                            str_detect(raw_name, "SE") ~ "SE",
                            str_detect(raw_name, "Adj") ~ "P_Adj",
                            str_detect(raw_name, "Count") ~ "Perm_Count")) %>%
  select(-raw_name) %>%
  pivot_wider(names_from = Metric, values_from = val) %>%
  mutate(CI_L = Beta - (1.96 * SE), 
         CI_U = Beta + (1.96 * SE),
         sig = ifelse(P_Adj < 0.05 & abs(Beta) > 0.2, "*", ""),
         # 1. Set factor levels correctly
         Bandwidth = factor(Bandwidth, levels = target_order),
         # 2. Create a unified group for consistent dodging/line paths
         DodgeGroup = interaction(Phase, Component)) %>%
  # 3. CRITICAL: Arrange data so lines connect in the correct X-axis order
  arrange(DodgeGroup, Bandwidth)

ggplot(plot_data, aes(x = Bandwidth, y = Beta, color = Phase, shape = Component, group = DodgeGroup)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -0.2, ymax = 0.2, fill = "gray85", alpha = 0.4) + 
  geom_hline(yintercept = 0, linetype = "dotted") +
  # Use the same DodgeGroup and width (0.6) for everything
  geom_errorbar(aes(ymin = CI_L, ymax = CI_U), width = 0.3, position = position_dodge(0.6)) +
  geom_line(position = position_dodge(0.6), alpha = 0.7) + 
  geom_point(size = 4, position = position_dodge(0.6)) +
  geom_text(aes(label = sig, y = CI_U + 0.1), position = position_dodge(0.6), size = 8, color = "black") +
  scale_color_manual(values = c("Phase II (Light)" = "#E69F00", "Phase III (Recovery)" = "#56B4E9")) +
  theme_bw() + 
  labs(title = "Acoustic Response (Corrected Line Paths)", 
       subtitle = "Lines now track correctly from low to high frequency bands.",
       caption = "Gray band = Null zone (|Beta| < 0.2).") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("Acoustic_Response_Final_Clean.pdf", width = 12, height = 8)

# =====================================================
# 6. MODEL DIAGNOSTICS
# =====================================================

library(MuMIn) # For R-squared (Marginal/Conditional)
library(car)   # For Levene's/Homoscedasticity checks

diagnostic_list <- list()

# Directory for plots
if(!dir.exists("Diagnostics")) dir.create("Diagnostics")

for(current_pc in pcs_to_model) {
  for(current_band in bands) {
    
    sub_data <- df_model %>% filter(Bandwidth == current_band)
    model_formula <- as.formula(paste(current_pc, "~ Treatment"))
    
    # Re-fit the observed model
    fit <- try(lme(fixed = model_formula, 
                   random = ~ 1 | Site/Exp_Date, 
                   correlation = corAR1(form = ~ 1 | Site/Exp_Date),
                   data = sub_data, control = lmeControl(opt = "optim")), silent = TRUE)
    
    if(inherits(fit, "try-error")) next
    
    # 1. R-Squared (Effect Size)
    # R2m = Fixed effects only | R2c = Fixed + Random effects
    r2 <- r.squaredGLMM(fit)
    
    # 2. Residual Normality (Shapiro-Wilk)
    resids <- residuals(fit, type = "normalized")
    shapiro_res <- shapiro.test(resids)
    
    # 3. Autocorrelation Check (Durbin-Watson equivalent for AR1)
    # We look at the ACF of the normalized residuals
    acf_val <- acf(resids, plot = FALSE)$acf[2] # Lag 1 autocorrelation
    
    # 4. Save Diagnostic Plots
    pdf(paste0("Diagnostics/Diag_", current_pc, "_", gsub(" ", "", current_band), ".pdf"), width = 10, height = 5)
    par(mfrow=c(1,3))
    # A. Residuals vs Fitted (Homoscedasticity)
    plot(fit, main = paste(current_pc, current_band, "Resids vs Fitted"))
    # B. QQ-Plot (Normality)
    qqnorm(resids); qqline(resids)
    # C. ACF Plot (Independence)
    acf(resids, main = "ACF of Normalized Residuals")
    dev.off()
    
    # Store Metrics
    diagnostic_list[[paste0(current_pc, "_", current_band)]] <- data.frame(
      Component = current_pc,
      Bandwidth = current_band,
      R2_Marginal = r2[1, "R2m"],
      R2_Conditional = r2[1, "R2c"],
      Shapiro_P = shapiro_res$p.value,
      Lag1_ACF = acf_val
    )
  }
}

Model_Diagnostics <- bind_rows(diagnostic_list)
print(Model_Diagnostics)
write.csv(Model_Diagnostics, "Model_Diagnostics_Summary.csv", row.names = FALSE)

# =====================================================
# 6. PC1 and PC2 Loadings & Variance Explained
# =====================================================

# 1. Extract Variance Explained
pca_importance <- summary(pca_res)$importance
pc1_var <- pca_importance[2, 1] * 100 # Proportion of Variance for PC1
pc2_var <- pca_importance[2, 2] * 100 # Proportion of Variance for PC2
cum_var <- pca_importance[3, 2] * 100 # Cumulative Proportion for PC1 + PC2

cat("\n--- PCA Variance Summary ---\n")
cat(sprintf("PC1 explains: %.2f%%\n", pc1_var))
cat(sprintf("PC2 explains: %.2f%%\n", pc2_var))
cat(sprintf("Total Variance Captured (PC1+PC2): %.2f%%\n", cum_var))
cat("----------------------------\n")

# 2. Extract the rotation (loadings) matrix
loadings_df <- as.data.frame(pca_res$rotation[, 1:2]) %>%
  mutate(Variable = rownames(.)) %>%
  rename(PC1_Loading = PC1, PC2_Loading = PC2) %>%
  mutate(Importance_PC1 = abs(PC1_Loading),
         Importance_PC2 = abs(PC2_Loading))

# 3. View the top drivers for each PC
top_pc1 <- loadings_df %>% arrange(desc(Importance_PC1)) %>% select(Variable, PC1_Loading) %>% head(10)
top_pc2 <- loadings_df %>% arrange(desc(Importance_PC2)) %>% select(Variable, PC2_Loading) %>% head(10)

print("--- Top Drivers for PC1 ---")
print(top_pc1)
print("--- Top Drivers for PC2 ---")
print(top_pc2)
# =====================================================
# 7. PCA SENSITIVITY CHECK: Variance Explained per Band
# =====================================================

band_variance_list <- list()

for(b in unique(df_model$Bandwidth)) {
  # Subset data for this specific band
  band_sub <- df_model %>% filter(Bandwidth == b)
  
  # Select the original scaled variables used for the PCA
  # (Using the variables that survived your findCorrelation filter)
  original_vars <- band_sub[, pca_vars]
  
  # Calculate Total Variance in this band
  total_var_band <- sum(apply(original_vars, 2, var, na.rm = TRUE))
  
  # Calculate Variance captured by Global PC1 and PC2 for this band
  pc_vars_band <- sum(var(band_sub$PC1, na.rm = TRUE), 
                      var(band_sub$PC2, na.rm = TRUE))
  
  # Proportion explained
  prop_explained <- pc_vars_band / total_var_band
  
  band_variance_list[[b]] <- data.frame(
    Bandwidth = b,
    Total_Band_Variance = total_var_band,
    Variance_Captured_by_Global_PCs = pc_vars_band,
    Percent_Explained = prop_explained * 100
  )
}

Band_PCA_Sensitivity <- bind_rows(band_variance_list)
print(Band_PCA_Sensitivity)

# Optional: Visualize the 'Deafness' of the PCA
ggplot(Band_PCA_Sensitivity, aes(x = Bandwidth, y = Percent_Explained, fill = Percent_Explained)) +
  geom_bar(stat = "identity") +
  geom_hline(yintercept = 10, linetype = "dashed", color = "red") +
  theme_bw() +
  labs(title = "Global PCA Sensitivity per Bandwidth",
       subtitle = "Percentage of band-specific variance captured by Global PC1 & PC2",
       y = "% Variance Explained",
       caption = "Values below 10% suggest the Global PCA is 'ignoring' this frequency range.") +
  scale_fill_gradient(low = "firebrick", high = "forestgreen")
```
