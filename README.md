# Synthetic Data Generation for Anomaly and Concept Drift co-Exploration

## References of this repository
- TSB-UAD (https://github.com/thedatumorg/TSB-UAD.git) for ECG data and NormA code
- DAMP (https://sites.google.com/view/discord-aware-matrix-profile/documentation) Matlab Code
- MOA (https://moa.cms.waikato.ac.nz), time-series drift generation


## References
- Will be apeared

## Contributors
- Will be apeared

## Installation

Steps:

1. Clone the repository git (need to change the repo name to CanGene)

```
git clone https://github.com/mac-dsl/AnomalyDriftDetection.git
```

2. Install dependencies from requirement.txt

```
pip install -r requirements.txt
```

## Benchmark
All datasets and time series are stored in ./data. We describe below the different types of datasets used in our benchmark.
1. ECG (./data/benchmark)
- The ECG dataset is a standard electrocardiogram dataset which comes from the MIT-BIH Arrhythmia database, where anomalies represent ventricular premature contractions. (Downloaded from https://github.com/TheDatumOrg/TSB-UAD/tree/main)
- Drift injected data for the experiments are stored under ./data/{dir}, dir=[test, test_mod]
  
2. Weather data
- The Weather dataset is a hourly, geographically aggregated temperature and radiation information in Europe originated from the NASA MERRA-2. The original Weather dataset contained timestamp, each country's temperature, and radiation information from 1960-to-2020, but for the demo, we separated each country's temperature data from 2017-to-2020 and saved them into '.arff' files in the repository. (From Open Power System Data, https://doi.org/10.25832/weather_data/2020-09-16:)
- Drift injected data for the experiments are stored under ./data/{dir}, dir=[test_weather, test_weather_mod]

## Usage
We include several demo. file to test the experiments.  

1. Weather_Test.ipynb
- This is the exactly same test file for the draft which is introduced in 'References'. All the parameters should be described in 'demo_config.yaml' file, so users can augment and change it based on purpose.
- This test inject anomalies for 3-different weather data (GR, NO, DE) and generate drifts between them. 
- Details would be described both in the paper and jupyter notebook.
  
2. anomaly_injection_demo.ipynb
- This notebook demonstrates applying CanGene for applying synthetic user-customized anomalies to a sample dataset.
- There are various parameters defined that allow the customization of user-defined anomalies, such as the type (point, collective, sequential) as well as their distrbutions and potential value ranges.
- Parameters are defined in demo_config.yaml

3.  ECG_drift_demo.ipynb
- This Notebook demonstrates applying CanGene for generating a sample data stream with concept drift between ECG signals of varying frequencies to imitate different heart rates.
- Parameters are defined in 'ECG_drift_demo_config.yaml'.
- 2 streams of ECG data are used, one of which is transformed to create the increased heart rate ECG signal.

4.  moa_drift_generation.ipynb
- This Notebook details the process of generating a drift between two source data streams without setting up a config file.
- More details are provided on the specific parameters used to generate drift as well as strategies for deciding on some parameters.
- This Notebook also shows how individual drift parameters can be viewed and manually updated to customize a generated stream post-generation.
