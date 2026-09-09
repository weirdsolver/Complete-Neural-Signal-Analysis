# Complete Neural Signal Analysis Toolbox

![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)
![Research Software](https://img.shields.io/badge/research-software-blue)
![No-Code GUI](https://img.shields.io/badge/no--code-GUI-brightgreen)
![EEG Analysis](https://img.shields.io/badge/EEG-analysis-purple)
![Neural Signal Processing](https://img.shields.io/badge/neural_signal-processing-orange)
![Human Cortical Organoids](https://img.shields.io/badge/human_cortical-organoids-teal)
![Open Source](https://img.shields.io/badge/open-source-success)

**Complete Neural Signal Analysis Toolbox** is an open-source research toolbox for EEG analysis, neural signal processing, electrophysiology, human cortical organoid signal analysis, multichannel time-series analysis, spectral analysis, entropy and complexity measures, nonlinear dynamics, fractal analysis, topology, graph theory, geometry, quantum-inspired methods, and neural-state manifold dynamics.

This repository is intended for researchers, developers, students, computational neuroscientists, neurotechnology builders, signal-processing engineers, brain-computer interface researchers, human cortical organoid researchers, government research groups, academic labs, and open-source scientific software users working with neural recordings, EEG data, organoid electrophysiology, multichannel time-series data, nonlinear neural dynamics, feature extraction, and advanced mathematical methods for neural systems.

Browse the folders to see the full method library and experimental tools.

---

## Current GUI / Web App Documentation

The current no-code GUI and integrated web application are documented here:

**[`GUI/neuro_signal_importer_analyzer/README.md`](GUI/neuro_signal_importer_analyzer/README.md)**

Start with that README if you want the latest drag-and-drop workflow, automatic installation checks, HTML/FastAPI frontend, neural-signal conversion backend, variable-electrode support, integrated NeuroMouse workbench, live replay, offline analysis, and comparison workflows.

Recommended launch:

```bash
cd GUI/neuro_signal_importer_analyzer
python3 run_neuro_signal_app.py
```

The launcher checks whether the software and required Python libraries are installed, installs missing dependencies only when needed, starts the local app server, and opens the browser interface.

---

## Quick Start

### Option 1: Use the No-Code GUI

Use the GUI folder above for the current end-to-end interface. It is designed for users who want to load neural recordings, convert files, analyze data, open NeuroMouse visualizations, run live replay, and compare datasets without writing code.

### Option 2: Browse the Toolbox Folders

Each folder contains methods for a different analysis area, including spectral analysis, entropy, nonlinear dynamics, fractal analysis, topology, geometry, quantum-inspired analysis, and neural-state manifold dynamics.

### Option 3: Use the Repository as Research Software

Researchers can use this repository as a reference toolbox, method library, educational resource, or starting point for reproducible neural signal analysis, EEG analysis, human cortical organoid signal analysis, electrophysiology workflows, computational neuroscience, and AI-assisted neural data analysis.

---

## GUI Capabilities

The GUI/web app documentation lives in `GUI/neuro_signal_importer_analyzer/README.md`. In brief, the current interface supports:

- Drag-and-drop neural signal files and folders
- Automatic installation and library checks
- Flexible conversion into a standardized neural-signal format
- Variable electrode/channel counts and channel names
- EEG, ECoG/iEEG-style, MEA-style, and human cortical organoid electrophysiology workflows
- Offline analysis and feature extraction
- Live replay of prerecorded recordings
- Group and dataset comparison workflows
- Integrated NeuroMouse workbench for interactive visualization
- Backend-generated NeuroMouse datasets instead of demo data
- Output folders with converted files, metadata, provenance, quality reports, and analysis products

---

## Supported Data and Input Types

The toolbox and GUI are designed around flexible neural signal workflows. Depending on the available modules and project setup, the repository can be used with:

- EEG files and recordings
- ECoG/iEEG-style recordings
- MEA and organoid electrophysiology recordings
- Human cortical organoid neural signal recordings
- Multichannel time-series matrices
- Folder-based datasets
- Table-based datasets
- Spreadsheet-style data layouts
- Custom neural signal data structures
- Feature tables and analysis exports
- Experimental neural recordings
- Data exported from external acquisition systems

---

## Main Research Areas

- EEG analysis
- Neural signal processing
- Human cortical organoid signal analysis
- Organoid electrophysiology
- Electrophysiology analysis
- Brain-signal analysis
- Multichannel time-series analysis
- Signal analysis and signal processing
- Spectral analysis and time-frequency analysis
- Entropy and complexity analysis
- Nonlinear dynamics and chaos
- Fractal and multifractal analysis
- Topological data analysis
- Graph theory and network neuroscience
- Geometry, manifolds, and curvature
- Quantum-inspired information analysis
- Neural-state and manifold dynamics
- AI-assisted neural data analysis
- Computational neuroscience
- Brain-computer interface research
- Neuroinformatics
- Reproducible scientific software

---

## Repository Structure

| Folder / Area | Description |
|---|---|
| `GUI/neuro_signal_importer_analyzer/` | Current no-code GUI/web app. See its README for detailed usage, installation, NeuroMouse integration, conversion, live replay, and comparison workflows. |
| `Data_Loading_and_Variable_Assigning_files/` | Data loading, variable assignment, signal preparation, and analysis setup. |
| `Dynamical_Systems/` | Dynamical systems, nonlinear dynamics, chaos, Lyapunov methods, attractor reconstruction, nonlinear coupling, and chaos control. |
| `Entropy/` | Entropy measures, complexity measures, multiscale entropy, sample entropy, approximate entropy, transfer entropy, entropy rate, and feature extraction. |
| `Fractal/` | Fractal analysis, fractal dimension, multifractal analysis, DFA, MFDFA, Hurst exponent, and wavelet-fractal analysis. |
| `Geometry/` | Manifolds, manifold learning, state-space geometry, Riemannian geometry, SPD manifolds, covariance manifolds, geodesics, curvature, and tangent-space analysis. |
| `Library/neural_signal_analysis/` | Core reusable neural signal analysis library code. |
| `Neural Net_files/` | Neural network, AI, machine learning, and computational modeling files. |
| `Quantum Analysis/` | Quantum-inspired, density-matrix, von Neumann entropy, quantum coherence, Hermitian matrix, and positive-semidefinite analysis. |
| `Spectral Analysis/` | Spectral analysis, PSD, Welch method, FFT, Lomb-Scargle, periodogram, wavelet transform, STFT, band power, coherence, and functional connectivity. |
| `Topology/` | Topology, topological data analysis, persistent homology, graph theory, weighted networks, modularity, small-world networks, graph metrics, and network neuroscience. |

---

## Suggested Uses

This toolbox may be useful for:

- EEG feature extraction
- Human cortical organoid research
- Human cortical organoid intelligence experiments
- Organoid electrophysiology analysis
- EEG biomarkers
- Neural signal analysis
- Brain-signal analysis
- Brain-computer interface signal processing
- BCI analysis
- Electrophysiology signal analysis
- AI for EEG and neural recordings
- Machine learning for neural signals
- Deep learning for EEG
- Nonlinear EEG analysis
- Neural complexity analysis
- Brain network analysis
- Functional connectivity analysis
- Time-frequency EEG analysis
- Riemannian EEG analysis
- Covariance manifold EEG analysis
- Topological neuroscience
- Fractal biomarkers
- Dynamical biomarkers
- Reproducible neural analysis
- Open-source computational neuroscience
- Research software indexing
- Academic software discovery
- No-code neural signal analysis
- GUI-based EEG analysis
- Flexible neural signal data structure support
- Beginner-friendly neural signal processing

---

## Links

- Current GUI README: [`GUI/neuro_signal_importer_analyzer/README.md`](GUI/neuro_signal_importer_analyzer/README.md)
- NeuroMouse software: https://github.com/UlaYuga/NeuroMouse
- Dataset: https://zenodo.org/records/15572614
- Repository: https://github.com/soulsyrup1/Complete-Neural-Signal-Analysis

---

## Keywords

The following terms describe the scope of this repository and support discovery for GitHub search, research software indexing, academic search, neuroscience software discovery, biomedical signal processing, AI research, government research, and open-source scientific software.

<details>
<summary>Core EEG / neural signal / time-series</summary>

## Core EEG / neural signal / time-series

eeg  
eeg-analysis  
neural-signal-processing  
signal-analysis  
signal-processing  
multichannel-time-series  
time-series-analysis  
nonlinear-time-series  
biomedical-signal-processing  
brain-signal-analysis  
electrophysiology-analysis  
human-cortical-organoid  
human-cortical-organoid-intelligence  
organoid-electrophysiology  
organoid-neural-signals  
analytic-signal  
hilbert-transform  
state-space-analysis  
phase-space-reconstruction  
delay-embedding  
attractor-reconstruction  

Additional SEO phrases:

EEG toolbox  
EEG processing toolbox  
EEG feature extraction  
EEG biomarkers  
neural data analysis  
brain data analysis  
human cortical organoid neural analysis  
human cortical organoid electrophysiology  
organoid intelligence signal analysis  
neurophysiology signal analysis  
computational neuroscience software  
open-source EEG analysis  
open-source neural signal processing  
brain-computer interface signal processing  
BCI analysis  
electrophysiology analysis  
neuroinformatics  
neural engineering  
brain dynamics  
neural time-series modeling  
biomedical AI  
AI for EEG  
machine learning for neural signals  
deep learning for EEG  
scientific Python neuroscience  
reproducible neural analysis  

</details>

<details>
<summary>Spectral / frequency / time-frequency</summary>

## Spectral / frequency / time-frequency

spectral-analysis  
power-spectral-density  
psd  
welch-method  
fast-fourier-transform  
fft  
lomb-scargle  
periodogram  
wavelet-transform  
continuous-wavelet-transform  
short-time-fourier-transform  
stft  
time-frequency-analysis  
band-power  
frequency-domain-analysis  
spectral-entropy  
spectral-centroid  
spectral-edge-frequency  
harmonics-detection  
phase-synchronization  
coherence-analysis  
functional-connectivity  

Additional SEO phrases:

EEG spectral analysis  
neural frequency analysis  
oscillatory brain activity  
alpha beta gamma theta delta bands  
neural oscillations  
time-frequency EEG  
wavelet EEG analysis  
Fourier neural signal analysis  
PSD brain signals  
coherence connectivity analysis  
phase-locking analysis  
cross-frequency analysis  
frequency-domain biomarkers  
neural synchrony  
brain network connectivity  
resting-state EEG analysis  
human cortical organoid spectral analysis  

</details>

<details>
<summary>Entropy / complexity</summary>

## Entropy / complexity

entropy  
entropy-measures  
complexity-measures  
complexity-science  
multiscale-entropy  
sample-entropy  
approximate-entropy  
transfer-entropy  
kolmogorov-sinai-entropy  
entropy-rate  
spectral-entropy  
feature-extraction  

Additional SEO phrases:

EEG entropy analysis  
neural complexity  
brain complexity metrics  
human cortical organoid complexity  
organoid neural complexity  
nonlinear complexity analysis  
physiological complexity  
complexity biomarkers  
entropy-based feature extraction  
information theory neuroscience  
signal irregularity  
neural variability  
multiscale brain dynamics  
complexity science for neuroscience  
biomedical entropy measures  
AI-ready neural features  

</details>

<details>
<summary>Dynamical systems / chaos / nonlinear methods</summary>

## Dynamical systems / chaos / nonlinear methods

dynamical-systems  
nonlinear-dynamics  
chaos  
chaos-theory  
lyapunov-exponents  
lyapunov-spectrum  
largest-lyapunov-exponent  
kaplan-yorke-dimension  
correlation-dimension  
false-nearest-neighbors  
surrogate-data  
iaaft  
rosenstein-method  
kantz-method  
wolf-algorithm  
benettin-algorithm  
convergent-cross-mapping  
ccm  
nonlinear-coupling  
chaos-control  
ogy-method  
pyragas-control  
arnold-tongues  
mode-locking  

Additional SEO phrases:

nonlinear EEG analysis  
chaotic neural dynamics  
chaotic organoid dynamics  
human cortical organoid dynamics  
neural attractors  
brain state-space analysis  
nonlinear brain modeling  
dynamical biomarkers  
phase-space EEG  
nonlinear neural coupling  
chaos in brain signals  
nonlinear systems neuroscience  
neural system identification  
computational dynamical neuroscience  
complex systems neuroscience  
dynamical systems toolbox  

</details>

<details>
<summary>Fractal / multifractal</summary>

## Fractal / multifractal

fractal-analysis  
fractal-dimension  
higuchi-fractal-dimension  
katz-fractal-dimension  
multifractal-analysis  
mfdfa  
detrended-fluctuation-analysis  
dfa  
hurst-exponent  
wavelet-fractal-analysis  

Additional SEO phrases:

EEG fractal dimension  
neural fractal analysis  
human cortical organoid fractal analysis  
multifractal EEG  
self-similarity brain signals  
scale-free neural dynamics  
long-range temporal correlations  
fractal biomarkers  
biomedical fractal analysis  
complexity and fractals  
Hurst exponent EEG  
DFA brain signals  
MFDFA neural signals  

</details>

<details>
<summary>Topology / TDA / graphs</summary>

## Topology / TDA / graphs

topology  
topological-data-analysis  
tda  
persistent-homology  
betti-numbers  
rips-complex  
network-science  
graph-theory  
graph-analysis  
weighted-networks  
community-detection  
modularity  
small-world-networks  
global-efficiency  
graph-metrics  
network-neuroscience  

Additional SEO phrases:

topological neuroscience  
TDA for EEG  
persistent homology brain signals  
topological biomarkers  
brain network analysis  
graph neural signals  
functional connectivity graphs  
neural graph metrics  
connectomics  
network neuroscience toolbox  
EEG network science  
topology of neural dynamics  
topological feature extraction  
graph-based biomarkers  
human cortical organoid network analysis  

</details>

<details>
<summary>Geometry / manifolds / curvature</summary>

## Geometry / manifolds / curvature

manifolds  
manifold-learning  
state-space-geometry  
intrinsic-geometry  
metric-geometry  
non-euclidean-geometry  
riemannian-geometry  
spd-manifold  
covariance-manifold  
affine-invariant-metric  
geodesics  
geodesic-interpolation  
geodesic-network  
riemann-curvature  
riemann-curvature-tensor  
ricci-curvature  
scalar-curvature  
sectional-curvature  
riemannian-barycenter  
parallel-transport  
curvature-analysis  
curvature-hotspots  
tangent-space  

Additional SEO phrases:

Riemannian EEG analysis  
covariance manifold EEG  
SPD matrix signal processing  
manifold-based neural analysis  
human cortical organoid manifold dynamics  
geometric deep learning neuroscience  
neural state geometry  
curvature of brain dynamics  
manifold biomarkers  
covariance geometry  
tangent-space EEG classification  
geodesic neural dynamics  
brain signal manifold learning  
intrinsic geometry of neural signals  

</details>

<details>
<summary>Quantum-like / information-geometric</summary>

## Quantum-like / information-geometric

quantum-inspired  
quantum-like  
quantum-information  
density-matrix  
von-neumann-entropy  
quantum-coherence  
effective-rank  
hermitian-matrices  
positive-semidefinite  

Additional SEO phrases:

quantum-inspired neural signal analysis  
density matrix EEG  
quantum-like cognition signals  
information geometry neuroscience  
von Neumann entropy brain signals  
matrix-based neural features  
positive semidefinite neural representations  
Hermitian matrix analysis  
effective rank for EEG  
quantum information methods for neural recordings  
human cortical organoid information geometry  

</details>

---

## Dataset

The dataset associated with this project is available here:

https://zenodo.org/records/15572614

This dataset can support reproducible testing, benchmarking, neural signal analysis experiments, EEG research, time-series analysis, human cortical organoid analysis workflows, and method development.

---

## Related Software

NeuroMouse software: https://github.com/UlaYuga/NeuroMouse

NeuroMouse is integrated into the current GUI/web application workflow. See the GUI README for current details:

[`GUI/neuro_signal_importer_analyzer/README.md`](GUI/neuro_signal_importer_analyzer/README.md)

---

## Citation and Indexing

Recommended project URL:

https://github.com/soulsyrup1/Complete-Neural-Signal-Analysis

Recommended dataset URL:

https://zenodo.org/records/15572614

Suggested citation text:

Complete Neural Signal Analysis Toolbox. Open-source neural signal analysis, EEG analysis, human cortical organoid signal analysis, electrophysiology analysis, nonlinear time-series analysis, spectral analysis, entropy analysis, fractal analysis, topology, geometry, manifold learning, and quantum-inspired analysis toolbox. https://github.com/soulsyrup1/Complete-Neural-Signal-Analysis

---

## License

**[Complete Neural Signal Analysis Toolkit](https://github.com/soulsyrup1/Complete-Neural-Signal-Analysis)** is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).

[![CC BY-SA 4.0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by-sa.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
