
# DMHALOMAPPER

_Transform Data, Accelerate Discovery, Unleash Innovation_

![Last Commit](https://img.shields.io/badge/last%20commit-today-brightgreen)
![Python](https://img.shields.io/badge/Python-80.0%25-orange)
![Languages](https://img.shields.io/badge/languages-2-blue)

> Built with the tools and technologies:  
> ![Markdown](https://img.shields.io/badge/-Markdown-000000?logo=markdown&logoColor=white)
> ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)
- [Features](#features)
- [Academic Context](#academic-context)
- [Citing This Work](#citing-this-work)
- [License](#license)

---

## Overview

**dmhalomapper** is a research-grade software toolkit developed to support complex data transformation, machine learning, and scientific visualization pipelines in astrophysics, particularly in cosmological structure formation studies. The package is tailored to unify diverse simulation and analysis stages into a reproducible and efficient pipeline.

### Why dmhalomapper?

This tool was built to streamline the development of predictive models for cosmic structures, with a strong emphasis on transparency, modularity, and scientific robustness. It is specifically optimized for dark matter halo mapping from EAGLE simulations using Recurrent Neural Network (RNN) architectures.

---

## Features

🌌 **Data Transformation & Mapping**  
> Integrates disparate astrophysical datasets, streamlines data alignment, and prepares them for modeling.

🔁 **End-to-End Pipeline**  
> Supports data loading, preprocessing, deep learning-based modeling, and high-quality visualization.

⚙️ **Hyperparameter Optimization**  
> Automates RNN tuning via Differential Evolution for optimal performance across diverse data regimes.

📈 **Visualization Tools**  
> Enables 2D/3D plotting of spatial structures, redshift evolution, and density profiles using Plotly.

🧩 **Modular & Extensible**  
> Fully extensible architecture designed to plug in new RNN variants, datasets, or astrophysical metrics.

📁 **Organized Output Management**  
> Ensures structured and traceable storage of logs, plots, CSVs, model checkpoints, and figures for publication.

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python ≥ 3.8
- **Package Manager:** Conda (recommended) or pip

### Installation

Build `dmhalomapper` from the source and install dependencies.

#### Clone the repository:

```bash
git clone https://github.com/Sazura/dmhalomapper
```

#### Navigate to the project directory:

```bash
cd dmhalomapper
```

#### Install the dependencies:

Using **conda**:

```bash
conda env create -f conda.yml
```

Or using **pip**:

```bash
pip install -r requirements.txt
```

---

### Usage

Run the project with:

Using **conda**:

```bash
conda activate (venv)
python (entrypoint)
```

The entrypoint can be `main.py` or a specific Jupyter notebook (e.g., `notebooks/DMHALO_pipeline.ipynb`), depending on your intended use.

---

### Testing

`dmhalomapper` uses the `pytest` test framework.

Using **conda**:

```bash
conda activate (venv)
pytest
```

---

## Academic Context

This project was developed as part of an undergraduate thesis at **Institut Teknologi Bandung (ITB)**. It aims to advance the use of machine learning in cosmological simulations, specifically focusing on the spatial mapping and temporal evolution of dark matter halos. It leverages the EAGLE DMONLY simulation dataset and deep learning tools to construct predictive models capable of capturing the hierarchical structure formation process in the universe.

Supervisor: **Dr. Muhamad Irfan Hakim**  
Author: **Sahabista Arkitanego Armantara** (Astronomy Department, ITB)

---

## Citing This Work

If you use `dmhalomapper` in your research, please cite:

> **Sahabista Arkitanego Armantara.**  
> *Pemetaan Dark Matter Halo Menggunakan Metode Recurrent Neural Network (RNN) dengan Data Simulasi EAGLE*.  
> Undergraduate Thesis, Institut Teknologi Bandung, 2024.

### BibTeX

```bibtex
@thesis{armantara2024dmhalomapper,
  author = {Sahabista Arkitanego Armantara},
  title = {Rekonstruksi Distribusi \textit{Dark Matter Halo} Menggunakan Metode Recurrent Neural Network (RNN) dengan Data Simulasi EAGLE},
  school = {Institut Teknologi Bandung},
  year = {2024},
  type = {Undergraduate Thesis}
}
```

---

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for more information.

---

📬 _For inquiries, collaboration opportunities, or contributions, please open an issue or reach out via email._
