# Seizure Onset Detection

![tests](https://github.com/tshibei/seizure-onset-detector/actions/workflows/test.yml/badge.svg)

Automated seizure onset detection on intracranial EEG, comparing classical
ML baselines (logistic regression, random forest) with a temporal CNN.

## Quickstart

```bash
git clone https://github.com/tshibei/seizure-onset-detector
cd seizure-onset-detector
uv sync --all-extras
uv run nbstripout --install   # strip notebook outputs on commit
uv run pytest
```

## Methods
### Data
 
We use the [SWEC-ETHZ long-term iEEG dataset](https://ieeg-swez.ethz.ch/),
which contains continuous intracranial EEG recordings from 18 patients
undergoing pre-surgical evaluation for drug-resistant epilepsy. Recordings
are provided as hourly `.mat` files numbered `_1h.mat` through `_Nh.mat`,
with per-patient annotation files (`IDxx_info.mat`) listing seizure onset
and offset times in seconds from recording start.
 
### Cohort selection
 
**Inclusion criteria.** A patient was included if both of the following held:
 
1. At least 3 distinct seizure events were annotated.
2. No more than 7 seizure annotations within any 4-hour window (excludes
   patients with annotation patterns consistent with status epilepticus or
   pipeline-level event subdivision).
Among kept patients, the densest 4-hour window contains at most 4 seizures;
among excluded patients, the densest contains at least 8. The threshold
cleanly separates the two groups.
 
**Excluded patients (n=8).**
 
| Patient | Seizures | Densest 4h window | Reason |
|---------|---------:|------------------:|--------|
| ID01    | 2        | 1                 | <3 seizures |
| ID02    | 2        | 1                 | <3 seizures |
| ID08    | 70       | 67                | Dense clustering (suspected status epilepticus) |
| ID09    | 27       | 8                 | Dense clustering |
| ID11    | 2        | 1                 | <3 seizures |
| ID14    | 60       | 34                | Dense clustering (suspected status epilepticus) |
| ID15    | 2        | 1                 | <3 seizures |
| ID17    | 2        | 1                 | <3 seizures |
 
For the three densely-clustered patients (ID08, ID09, ID14), the
annotation pattern is consistent with either status epilepticus or an
annotation pipeline that subdivided a single prolonged event into many
short sub-events. Treating these as independent training examples would
inflate per-patient seizure counts without contributing independent
information, and would also compromise leave-one-patient-out evaluation
by giving a small number of patients disproportionate weight.

For each included patient, we additionally include 4 interictal hours
distributed across the recording to provide clean negative examples and
support false-positives-per-hour estimation.

### Final cohort
We use 10 patients from the [SWEC-ETHZ iEEG dataset](https://ieeg-swez.ethz.ch/),
covering sampling rates of 512–1024 Hz and 32–128 implanted channels. Across 10 patients, we have 4–17 seizures each (median 6), with channel counts of 32–128 and sampling rates of 512 or 1024 Hz. 

| Patient   |   Seizures |   Sampling Rate (Hz) |   Channels |   Downloaded (h) |   Ictal (%) |
|:----------|-----------:|---------------------:|-----------:|-----------------:|------------:|
| ID03      |          4 |                  512 |         64 |                8 |         0.9 |
| ID04      |         14 |                 1024 |         32 |               17 |         1.0 |
| ID05      |          4 |                  512 |        128 |                9 |         0.2 |
| ID06      |          8 |                 1024 |         32 |               12 |         0.8 |
| ID07      |          4 |                  512 |         75 |                8 |         1.0 |
| ID10      |         17 |                 1024 |         32 |               20 |         1.7 |
| ID12      |          9 |                 1024 |         56 |               13 |         2.8 |
| ID13      |          7 |                 1024 |         64 |               11 |         1.8 |
| ID16      |          5 |                 1024 |         34 |                9 |         2.9 |
| ID18      |          5 |                 1024 |         42 |                9 |         3.1 |

*Recordings were sub-sampled around seizure events, so the ictal fraction
shown here substantially exceeds what would be observed in continuous
monitoring. FPR/hour reported in Results should be interpreted with this
in mind.* 

![Seizures per patient](figures/seizures_per_patient.png)

*Seizure occurrence across downloaded recordings for each patient. Files are
labeled by recording hour (relative to monitoring start). Most patients have
1–2 seizures per affected file; ID04 (15h) and ID10 (18h) each contain two.*

![Seizure duration](figures/seizure_duration_histogram.png)

*Most seizures are <90s.*

### Reproducing the cohort
 
The cohort and per-patient file lists are defined declaratively in
`scripts/download_swec.py`, which reads each patient's `_info.mat` file
to derive the relevant hourly files. Note that seizure onset/offset times
in the info files are reported in seconds from recording start
(0-indexed), while hourly files are named with 1-indexed hour numbers —
a seizure starting at 297,000 seconds (82.5h) is contained in `_83h.mat`.
 
To reproduce the cohort:
 
```bash
uv run python scripts/download_swec.py
```
 


## Status

Work in progress — see `notebooks/` for current results.

## Results

TBD (Day 3+).

## Limitations

- We sub-sampled the long-term recordings, prioritizing files containing seizures plus interictal context. Ictal fraction in the downloaded subset is therefore much higher than in the full continuous recording.  
- FPR/hour is computed on a sub-sampled interictal set rather than continuous monitoring; deployment FPR would likely be higher.