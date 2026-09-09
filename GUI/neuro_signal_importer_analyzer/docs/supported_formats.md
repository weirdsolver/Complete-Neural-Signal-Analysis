# Supported format families

The importer prioritizes continuous neural signals. Supported families include:

- DSamp-style MATLAB files
- EEGLAB-style MATLAB files
- FieldTrip-style MATLAB files
- generic MATLAB arrays
- EDF/BDF through optional MNE
- CSV/TSV/Excel signal tables
- NPY/NPZ arrays
- generic HDF5 continuous signal files
- FinalSpark LiveMEA-style HDF5 files
- Cortical Labs CL1-style HDF5 files
- NWB-like/HDF5 continuous arrays
- MaxWell/HD-MEA-like HDF5 arrays when a continuous candidate can be found

Unsupported by design in this project version:

- spike-only datasets
- stimulus/event logic
- behavior/task annotations
- stimulation amplitude/frequency/location/block outputs
