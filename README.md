# **Reverse Game of Life with CNN**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10.19](https://img.shields.io/badge/Python-3.10.19-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626.svg?logo=Jupyter&logoColor=white)](https://jupyter.org/)

## 📑 Overview
The Game of Life (GoL) is a cellular automaton characterised by non-linear evolution and emergent complexity. Its global state
transition function is non-injective and irreversible, leading to information loss. Consequently, the Reverse GoL, i.e., finding a predecessor that
evolves into a given target after a given number of generations, is an NP-complete task. In this paper, we introduce a differentiable GoL transition function within a convolutional neural network-based model to
reconstruct the probability distribution, i.e. a heatmap, of a possible initial state associated with the given final board. In this study, the
models are validated on 15×15 boards after one generation by analysing structure-based metrics on the heatmaps of the predicted initial states. In
particular, we computed the fuzziness index to measure the degree of binarisation, the Earth Mover’s Distance, to evaluate the accuracy of the
spatial mass distribution, and the percentage of high uncertainty cells within a range, to quantify prediction confidence. Our results demonstrate that integrating the differentiable layer reduces the fuzziness index
by approximately 40% compared to the baseline approach. Furthermore, the analysis indicate that pixel-wise metrics, such as Mean Squared Error, can be misleading in this context, as they ignore the spatial context
of cells. In contrast, the use of structural metrics reveals that the proposed architecture effectively captures the underlying physics and spatial
organisation of the automaton.

## 🏗️ Repository Structure

The project is organized into modular directories separating the research environment from the execution pipeline:

* **`Auxiliares/`**: Core helper functions, custom classes, and shared utilities utilized across the execution scripts.
* **`Crossvalidation/`**: Scripts dedicated specifically to the cross-validation procedures for the Classic model.
* **`Notebooks/`**: Jupyter notebooks containing exploratory data analysis (EDA), experimental results, and detailed research findings.
* **`TrainVal/`**: Execution scripts handling the model training and validation phases.
* **`Test/`**: Scripts for final model testing and performance evaluation on hold-out data.

## ⚠️ Prerequisites
Ensure you have the following installed before running the pipeline:
* Python 3.10.x
* keras==3.12.0
* keras_tuner==1.4.8
* matplotlib==3.10.8
* numpy==2.4.2
* pandas==3.0.1
* pot==0.9.6.post1
* sacred==0.8.7
* scipy==1.17.1
* tensorflow==2.20.0

You can install these dependences by running: 
`1pip install -r requirements.txt`


## 🔢 Usage Pipeline
To reproduce the research or evaluate the models, follow this standard execution flow:

1. **Cross-Validation (Classic Model):** Run the scripts in `Crossvalidation/` to perform robust validation specifically on the classic architecture.
2. **Training & Validation:** Execute the relevant scripts within the `TrainVal/` directory to train the models and tune hyperparameters.
3. **Testing:** Once trained, evaluate the models against the test dataset using the scripts in the `Test/` directory.
4. **Analysis:** Review the files in `Notebooks/` for visual representations of metrics, data distributions, and final research conclusions.

## License

This project is distributed under the [GPL-3.0 License](LICENSE). See the `LICENSE` file for more details.
