# EEG-CASCADE: Personalized EEG Seizure Detection and Early Warning

## Overview

This repository contains the implementation of **EEG-CASCADE**, a patient-specific forecasting-based anomaly detection framework for seizure-onset detection and preictal early warning from multichannel EEG.

EEG-CASCADE extends the conceptual anomaly-detection strategy of CASCADE to continuous multichannel EEG using **Dynamical Systems Machine Learning (DynML)** with Rössler dynamical reservoirs. Rather than training a supervised classifier using ictal or preictal labels, the framework learns only clean patient-specific interictal dynamics and identifies seizure-associated states through persistent degradation of one-step forecasting likelihood.

The repository accompanies the manuscript:

**Signals Before Seizure: Personalized AI Early Warning of Ictal Onset**

---

## Software environment

The analyses were performed on NIH Biowulf using Python 3.10.14.
The recorded software versions are:

| Software | Version |
|----------|---------|
| Python | 3.10.14 |
| NumPy | 2.2.6 |
| SciPy | 1.15.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| Matplotlib | 3.9.1 |
| MNE-Python | 1.12.1 |
| JupyterLab | 4.5.7 |
| Notebook | 7.5.6 |

Batch experiments used NIH Biowulf's Swarm job-submission system.
Individual analysis scripts can also be run directly with Python;
Swarm is required only for the supplied cluster submission commands.
Paths in the generators and notebooks must be adjusted to the local
code, data, and results directories.

---

# Repository Organization

## Core EEG-CASCADE (DynML) Models

Three patient-specific analyses are included in the study.

### CHB-MIT Seizure-Onset Detection

- `chbmit_dynml_ictal_detection_BD_v4_fixed.py`

Main EEG-CASCADE implementation for seizure-onset detection in the CHB-MIT scalp EEG database.

The analysis includes:

- CHB-MIT EEG loading and preprocessing
- Common 18-channel longitudinal bipolar montage
- 0.5–40 Hz band-pass filtering
- 30-s seizure-onset test windows
- Clean interictal sequence construction
- Patient-specific training, validation, and held-out testing
- Multichannel EEG feature extraction
- Training-only standardization and PCA
- Rössler DynML forecasting
- Validation-derived prediction-likelihood anomaly scoring
- Persistence-based seizure alarm generation
- Seizure-onset sensitivity, specificity, balanced accuracy, false alarms per hour, and detection delay
- Empirical finite-time Bowen–Dinaburg entropy analysis

### CHB-MIT Preictal Early Warning

- `chbmit_dynml_preictal_35to5_BD_v4.py`

Main EEG-CASCADE implementation for patient-specific preictal early warning using the predefined interval from **35 to 5 minutes before seizure onset**.

The analysis includes:

- 5-min seizure prediction horizon (SPH)
- 30-min seizure occurrence period (SOP)
- 15-s EEG chunks with a 5-s stride
- 30-min clean interictal sequences
- Patient-specific training, validation, and held-out testing
- Multichannel EEG feature extraction
- Training-only standardization and PCA
- Rössler DynML forecasting
- Validation-derived anomaly scoring
- 60-s persistence-confirmed early-warning alarms
- Preictal sensitivity, specificity, balanced accuracy, false alarms per hour, and warning lead time
- Empirical finite-time Bowen–Dinaburg entropy analysis

### TUSZ External Seizure-Onset Detection

- `tusz_dynml_ictal_detection_BD_v4.py`

EEG-CASCADE implementation for external seizure-onset detection using version 2.0.6 of the Temple University Hospital Seizure Corpus (TUSZ).

The TUSZ analysis uses the same forecasting-based anomaly-detection principle as the CHB-MIT seizure-onset analysis while adapting EEG preprocessing to the TUSZ recording structure and common 16-channel bipolar montage.

The analysis includes:

- TUSZ EEG and seizure-annotation loading
- Duplicate seizure-event annotation handling
- Common 16-channel longitudinal bipolar montage
- Reconstruction of bipolar derivations from referential channels when required
- Patient-specific interictal training, validation, and held-out testing
- 30-s seizure-onset test windows
- Multichannel EEG feature extraction
- Training-only standardization and PCA
- Rössler DynML forecasting
- Validation-derived anomaly scoring
- Persistence-based seizure detection
- External evaluation of sensitivity, specificity, balanced accuracy, false alarms per hour, and detection delay

---

## Data Generation and Experiment Scripts

The following scripts generate the swarm jobs used for the parameter-regime and reservoir-seed experiments reported in the manuscript.

### CHB-MIT Seizure-Onset Detection

- `generate_swarm_chbmit_ictal_BD_v4_fixed_all_patients.py`

Generates the patient, Rössler-regime, and reservoir-seed combinations for the CHB-MIT seizure-onset analysis.

### CHB-MIT Preictal Early Warning

- `generate_swarm_chbmit_preictal_35to5_BD_v4.py`

Generates the patient, Rössler-regime, and reservoir-seed combinations for the CHB-MIT 35-to-5-min preictal analysis.

### TUSZ Seizure-Onset Detection

- `generate_swarm_tusz_ictal_BD_v4.py`

Generates the patient, Rössler-regime, and reservoir-seed combinations for the TUSZ external seizure-onset analysis.

---

## TUSZ Patient Selection

- `TUSZ_patients_selection.ipynb`

Used to identify patients satisfying the predefined TUSZ eligibility criteria for the external seizure-onset analysis.

Eligible patients were required to have sufficient seizure-onset windows and patient-specific clean interictal data for training, validation, and held-out testing.

The predefined selection procedure yielded **63 eligible TUSZ patients** before model-specific numerical integration failures were considered.

---

## Figure Generation Notebooks

Each notebook reproduces one or more main-text or supplementary figures from the manuscript.

| Notebook | Description | Output Figures |
|----------|-------------|----------------|
| `Fig_4,5,S3,S4_plot_chbmit_ictal_BD_results.ipynb` | CHB-MIT seizure-onset performance, patient-level results, and Bowen–Dinaburg entropy analyses | Figures 4, 5; Supplementary Figures S3, S4 |
| `Fig_6,7,S5,S6_plot_chbmit_preictal_35to5_BD_results.ipynb` | CHB-MIT 35-to-5-min preictal early-warning performance and Bowen–Dinaburg entropy analyses | Figures 6, 7; Supplementary Figures S5, S6 |
| `Fig_8,9_plot_tusz_ictal_results.ipynb` | TUSZ external seizure-onset detection performance | Figures 8, 9 |
| `Fig_S1_CHBMIT_seizure_distribution.ipynb` | CHB-MIT seizure-duration distribution used to characterize seizure-onset window selection | Supplementary Figure S1 |
| `Fig_S1_TUSZ_seizure_distribution.ipynb` | TUSZ seizure-duration distribution used to characterize seizure-onset window selection | Supplementary Figure S1 |
| `Fig_S2_pca_explained_variance_training_interictal_PCA30.ipynb` | Patient-specific training-interictal PCA explained-variance analysis for CHB-MIT and TUSZ | Supplementary Figure S2 |

> **Note:** Main-text Figures **1**, **2**, and **3** are conceptual/methodological schematics and therefore do not have corresponding computational plotting notebooks. All data-derived main-text and supplementary figures are reproduced using the notebooks listed above.

---

# EEG-CASCADE Analysis Pipeline

The typical workflow is:

1. Obtain the CHB-MIT and/or TUSZ EEG datasets from their original repositories.
2. Parse seizure annotations and construct the patient-specific bipolar EEG montage.
3. Band-pass filter EEG signals from 0.5 to 40 Hz.
4. Construct clean interictal, seizure-onset, or preictal sequences according to the experiment-specific temporal definitions.
5. Extract predefined multichannel EEG features.
6. Fit patient-specific standardization and PCA using only training-interictal EEG.
7. Generate the required patient × Rössler-regime × reservoir-seed jobs using the corresponding swarm generator.
8. Train the DynML readout using only clean interictal transitions.
9. Forecast the subsequent EEG feature state using the Rössler dynamical reservoir.
10. Estimate the validation-interictal residual distribution.
11. Compute cumulative forecast log likelihood and validation-derived anomaly thresholds.
12. Apply the experiment-specific persistence criterion to generate confirmed alarms.
13. Calculate seizure-onset or preictal performance metrics.
14. Compute empirical finite-time Bowen–Dinaburg entropy for the Rössler reservoirs.
15. Generate manuscript figures and statistical summaries using the corresponding notebooks.

---

# EEG Feature Representation

Each EEG analysis chunk is represented using **15 predefined features per channel** spanning time-domain, dynamical, and spectral characteristics:

- Root-mean-square amplitude
- Line length
- Peak-to-peak amplitude
- Hjorth activity
- Hjorth mobility
- Hjorth complexity
- Log delta-band power
- Log theta-band power
- Log alpha-band power
- Log beta-band power
- Log gamma-band power
- Theta-to-beta power ratio
- Alpha-to-beta power ratio
- Spectral entropy
- Zero-crossing rate

Concatenation across channels yields:

- **270-dimensional feature vectors** for the 18-channel CHB-MIT montage
- **240-dimensional feature vectors** for the 16-channel TUSZ montage

Patient-specific standardization and PCA are fitted exclusively using training-interictal data. Up to 30 principal components are retained for DynML forecasting.

---

# DynML / Rössler Forecasting

EEG-CASCADE performs one-step forecasting of the patient-specific PCA feature state:

`x_t -> x_(t+1)`

Each input state is mapped into the initial conditions of a dynamical reservoir. The sampled-window analyses used **10 independent Rössler systems**. The continuous analysis additionally evaluated reservoir counts of **5, 10, 20, 30, 40, and 50**.

The resulting reservoir state is used by a ridge-regularized linear readout to forecast the subsequent EEG feature state.

Only the linear readout is trained. The input projection and Rössler reservoir parameters remain fixed after initialization.

The sampled-window regime comparisons evaluated **12 parameter regimes (R0–R11)**. The continuous analysis evaluated **23 regimes (R0–R22)** across the six reservoir counts. Five reservoir realizations used seeds **1107, 1249, 1388, 1523, and 1697**, with EEG data seed **17**.

---

# Forecast-Likelihood Anomaly Detection

EEG-CASCADE identifies seizure-associated states through degradation of forecasting likelihood relative to patient-specific interictal dynamics.

The procedure consists of:

1. Compute one-step forecast residuals.
2. Fit a diagonal Gaussian residual distribution using validation-interictal transitions.
3. Convert forecast residuals into transition log likelihoods.
4. Accumulate transition log likelihood sequentially within each EEG sequence.
5. Define a position-dependent anomaly threshold using the fifth percentile of validation-interictal cumulative likelihoods.
6. Identify raw anomalies when cumulative likelihood falls below the corresponding validation threshold.
7. Confirm an alarm only when the anomaly condition persists continuously for the predefined duration.

Persistence criteria used in the study were:

- **3 s and 5 s** for seizure-onset detection
- **60 s** for the primary CHB-MIT preictal early-warning analysis

## Continuous-recording evaluation

Continuous seizure-onset evaluation used held-out CHB-MIT recordings.
All seizure-containing recordings and a subset of seizure-free recordings
were reserved for testing, with training and validation sequences drawn
from the remaining recordings.

Continuous scoring used a rolling log-likelihood sum matching the
30-s validation sequences and a scalar fifth-percentile validation
threshold. Recordings were processed independently without concatenation.
One alarm was counted per contiguous anomalous episode after persistence
confirmation. False-alarm burden was calculated over interictal recording
time outside the predefined peri-seizure exclusion intervals.

The primary continuous configuration was R4 with N=10 and 3-s persistence.
Exploratory BD analyses evaluated R0–R22 across N=5, 10, 20, 30, 40, and 50.

---

# Bowen–Dinaburg Reservoir Analysis

The repository also implements an empirical finite-time Bowen–Dinaburg separated-orbit entropy statistic to characterize the dynamical richness of the Rössler reservoirs.

The analysis uses training-interictal reservoir trajectories only and is independent of seizure outcomes.

The principal parameters are:

- Q = 64 sampled training-interictal states
- epsilon = 0.05
- n = 30 sampled time points
- temporal sampling interval = 0.05
- 25 randomized greedy separated-set repetitions

The statistic is interpreted as a **finite-sample measure of reservoir trajectory distinguishability**, rather than as an exact estimate of asymptotic topological entropy.

Bowen–Dinaburg entropy is used as a dynamical diagnostic and not as an outcome-driven criterion for selecting reservoir regimes.

---

# Performance Metrics

The analyses report patient-specific and cohort-level measures including:

- Seizure-onset sensitivity
- Preictal early-warning sensitivity
- Interictal specificity
- Balanced accuracy
- False alarms per hour of held-out interictal EEG
- Persistence-confirmed seizure detection delay
- Persistence-confirmed preictal warning lead time

Patient-level results are summarized across independent reservoir realizations.

Relationships between empirical finite-time Bowen–Dinaburg entropy and detection performance are evaluated using Spearman rank correlation.

---

# Datasets

Experiments were performed using:

### CHB-MIT Scalp EEG Database

Used for:

- Patient-specific seizure-onset detection
- Patient-specific preictal early warning
- Reservoir dynamical analysis

The CHB-MIT data are publicly available through PhysioNet.

### Temple University Hospital Seizure Corpus (TUSZ), Version 2.0.6

Used for:

- External patient-specific seizure-onset detection
- Evaluation of EEG-CASCADE in a more heterogeneous clinical scalp-EEG cohort

Please obtain both datasets from their original repositories. Raw EEG data are not distributed with this repository.

---

# Main Experimental Configurations

## CHB-MIT Seizure-Onset Detection

- Positive interval: seizure onset to +30 s
- EEG chunk length: 5 s
- Chunk stride: 1 s
- Clean interictal exclusion: 60 min before seizure onset through 60 min after seizure end
- Interictal sequence duration: 30 s
- Interictal selection stride: 30 s
- Maximum interictal sequences per patient: 120
- Approximate split: 50% training / 25% validation / 25% held-out testing
- Primary Rössler configuration: R4
- Primary persistence: 3 s
- Secondary persistence: 5 s

## CHB-MIT Preictal Early Warning

- Preictal interval: -35 to -5 min
- Seizure prediction horizon: 5 min
- Seizure occurrence period: 30 min
- EEG chunk length: 15 s
- Chunk stride: 5 s
- Interictal sequence duration: 30 min
- Interictal selection stride: 30 min (nonoverlapping sequences)
- An overlap audit confirmed no shared EEG samples across the selected training, validation, and held-out test subsets.
- Maximum interictal sequences per patient: 120
- Approximate split: 50% training / 25% validation / 25% held-out testing
- Primary Rössler configuration: R4
- Alarm persistence: 60 s

## TUSZ Seizure-Onset Detection

- Positive interval: seizure onset to +30 s
- EEG chunk length: 5 s
- Chunk stride: 1 s
- Clean interictal exclusion: 2 min before seizure onset through 2 min after seizure end
- Interictal sequence duration: 30 s
- Interictal selection stride: 60 s
- Maximum interictal sequences per patient: 200
- Approximate split: 50% training / 25% validation / remaining approximately 25% held-out testing
- Primary Rössler configuration: R4
- Primary persistence: 3 s
- Secondary persistence: 5 s

---

# Features

- Personalized patient-specific EEG modeling
- Training using only clean interictal EEG
- No supervised ictal or preictal examples required for model fitting
- Multichannel EEG feature-state forecasting
- Dynamical Systems Machine Learning (DynML)
- Rössler dynamical reservoirs
- Training-only standardization and PCA
- Validation-calibrated probabilistic anomaly detection
- Persistence-based alarm confirmation
- Seizure-onset detection
- Preictal early warning
- External evaluation using TUSZ
- Empirical finite-time Bowen–Dinaburg reservoir analysis
- Patient-specific and reservoir-specific uncertainty characterization

---

# Repository Files

The principal files included in this repository are:

```text
chbmit_dynml_ictal_detection_BD_v4_fixed.py
chbmit_dynml_preictal_35to5_BD_v4.py
tusz_dynml_ictal_detection_BD_v4.py

generate_swarm_chbmit_ictal_BD_v4_fixed_all_patients.py
generate_swarm_chbmit_preictal_35to5_BD_v4.py
generate_swarm_tusz_ictal_BD_v4.py

TUSZ_patients_selection.ipynb

Fig_4,5,S3,S4_plot_chbmit_ictal_BD_results.ipynb
Fig_6,7,S5,S6_plot_chbmit_preictal_35to5_BD_results.ipynb
Fig_8,9_plot_tusz_ictal_results.ipynb
Fig_S1_CHBMIT_seizure_distribution.ipynb
Fig_S1_TUSZ_seizure_distribution.ipynb
Fig_S2_pca_explained_variance_training_interictal_PCA30.ipynb
