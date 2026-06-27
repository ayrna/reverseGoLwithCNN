# **Reverse Game of Life with CNN**
![Status](https://img.shields.io/badge/status-stable-8a2be2?logo=git&logoColor=white)
[![Python 3.10.19](https://img.shields.io/badge/Python-3.10.19-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626.svg?logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## 📑 Overview
### **[A CNN-based approach to reverse Game of Life](https://link.springer.com/chapter/10.1007/978-3-032-27317-8_25)**
**Abstract:** The Game of Life (GoL) is a cellular automaton characterised by non-linear evolution and emergent complexity. Its global state
transition function is non-injective and irreversible, leading to information loss. Consequently, the Reverse GoL, i.e., finding a predecessor that
evolves into a given target after a given number of generations, is an NP-complete task. In this paper, we introduce a differentiable GoL transition function within a convolutional neural network-based model to
reconstruct the probability distribution, i.e. a heatmap, of a possible initial state associated with the given final board. In this study, the
models are validated on 15×15 boards after one generation by analysing structure-based metrics on the heatmaps of the predicted initial states. In
particular, we computed the fuzziness index to measure the degree of binarisation, the Earth Mover’s Distance, to evaluate the accuracy of the
spatial mass distribution, and the percentage of high uncertainty cells within a range, to quantify prediction confidence. Our results demonstrate that integrating the differentiable layer reduces the fuzziness index
by approximately 40% compared to the baseline approach. Furthermore, the analysis indicate that pixel-wise metrics, such as Mean Squared Error, can be misleading in this context, as they ignore the spatial context
of cells. In contrast, the use of structural metrics reveals that the proposed architecture effectively captures the underlying physics and spatial
organisation of the automaton.

--- 
### **Ablation of the Differentiable Game of Life for the reverse problem**
**Abstract:** Recent advancements in the reverse Game of Life problem
have demonstrated the efficacy of integrating a Differentiable Game of
Life layer within Convolutional Neural Networks along with a fuzziness
index penalty (the DiffGoL methodology). However, the exact contri-
bution of each individual component to the overall performance of the
model remains unquantified, raising questions about the necessity and
impact of each element. In this paper, we present a comprehensive ab-
lation study of the DiffGoL methodology by evaluating the performance
of the model under various configurations where specific components are
systematically removed. These variants are validated on 15 × 15 states
after one generation by analysing structure-based metrics along with the
Mean Squared Error of the predicted heatmaps of the initial and final
states. Validation criteria prioritise evolutionary consistency toward the
final states and the structural fidelity of the initial heatmaps directly
against the ground-truth, without relying on intermediate binarization
thresholds. Our findings reveal that no single component is solely respon-
sible for the model’s performance. Instead, the absence of the diffGoL
layer leads to evolutionary inconsistency, whereas removing the fuzzi-
ness index penalty results in severe binarization failure. This highlights
the strict synergy between the architectural components, establishing a
robust framework for mapping continuous neural representations to dis-
crete cellular automata spaces.

## 📂 Repository Structure

The project is organized into modular directories separating the research environment from the execution pipeline:

<details>
<summary><strong>TensorFlow</strong></summary>
  
* **`Auxiliares/`**: Core helper functions (`functions`), custom classes (`Clases`) utilized across the execution scripts.
* **`Crossvalidation/`**: Scripts dedicated specifically to the cross-validation procedures for the Classic model.
* **`Notebooks/`**: Jupyter notebooks containing exploratory data analysis (EDA), experimental results, and detailed research findings.
* **`TrainVal/`**: Execution scripts handling the model training and validation phases.
* **`Test/`**: Scripts for final model testing and performance evaluation on hold-out data.
</details>

<details>
<summary><strong>PyTorch</strong></summary>
  
* **`Notebooks_15x15/`**: Jupyter notebooks containing exploratory data analysis (EDA), experimental results, and detailed research findings for 15x15 boards.
* **`Notebooks_20x20/`**: Jupyter notebooks containing exploratory data analysis (EDA), experimental results, and detailed research findings for 20x20 boards.
* **`Notebooks_25x25/`**: Jupyter notebooks containing exploratory data analysis (EDA), experimental results, and detailed research findings for 25x25 boards.
* **`TrainTest/`**: Execution scripts handling the model training, validation and test phases.
* **`utils/`**: Core helper functions and custom classes utilized across the execution scripts.

</details>

<details>
<summary><strong>PostProcessing</strong></summary>
  
* **`GenerateDataset/`**: Execution scripts handling the data generation.
* **`utils/`**: Core helper functions and custom classes utilized across the execution scripts.

</details>


## 🛠️ Prerequisites
Ensure you have the following installed before running the pipeline:
<details>
<summary><strong>TensorFlow</strong></summary>

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
</details>

<details>
<summary><strong>PyTorch</strong></summary>

* Python 3.10.x
* joblib==1.5.3
*matplotlib==3.10.9
* numpy==2.4.4
* pandas==3.0.3
* pot==0.9.6.post1
* sacred==0.8.7
* scikit_learn==1.8.0
* scipy==1.17.1
* torch==2.11.0+cu126
* torchmetrics==1.9.0
</details>

These dependences can be installed by running:  
`pip install -r requirements_tf.txt`
`pip install -r requirements_torch.txt`


## 🔬 Usage Pipeline
To reproduce the research or evaluate the models, follow this standard execution flow:

<details>
<summary><strong>TensorFlow</strong></summary>

1. **Cross-Validation (Classic Model):** Run the scripts in `Crossvalidation/` to perform robust validation specifically on the classic architecture.
2. **Training & Validation:** Execute the relevant scripts within the `TrainVal/` directory to train the models and tune hyperparameters.
3. **Testing:** Once trained, evaluate the models against the test dataset using the scripts in the `Test/` directory.
4. **Analysis:** Review the files in `Notebooks/` for visual representations of metrics, data distributions, and final research conclusions.

</details>

<details>
<summary><strong>Torch</strong></summary>

1. **Training, validation & Test:** Run the scripts in `TrainTest/` to perform the training, validation and testing of the models.
2. **Analysis:** Review the files in `Notebooks/` for visual representations of metrics, data distributions, and final research conclusions.

</details>

## ⚖️ License

This project is distributed under the [GPL-3.0 License](LICENSE). See the `LICENSE` file for more details.
