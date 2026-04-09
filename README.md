# Quantum Machine Learning: Variational Quantum Classifier on make_moons

A beginner-friendly Quantum Machine Learning project implementing a variational quantum classifier using PennyLane, with a direct comparison to a classical logistic regression model on a nonlinear dataset.


## Overview

This project explores the basics of Quantum Machine Learning (QML) by building a hybrid quantum-classical classifier. A variational quantum circuit is used to classify data points from a nonlinear dataset and its performance is compared against a classical machine learning model.


## Objective

The goal of this project is to understand:

* How classical data can be encoded into quantum circuits
* How variational quantum models are trained
* How quantum models perform compared to classical baselines


## Tech Stack

* Python
* PennyLane
* scikit-learn
* NumPy
* matplotlib
* Jupyter Notebook


## Workflow

1. Generate the make_moons dataset
2. Split and scale the data
3. Train a classical logistic regression model
4. Build a 2-qubit variational quantum circuit
5. Train the quantum classifier using gradient descent
6. Compare model performance


## Results

* Classical Model Accuracy: **0.80**
* Quantum Model Accuracy: **0.52**

### Interpretation

The classical model outperforms the quantum classifier in this setup. This is expected due to the limited size and depth of the quantum circuit. The project highlights how quantum models are implemented and evaluated rather than outperforming classical methods.


## Visualizations

### Dataset

![Dataset](results/dataset_plot.png)

### Training Loss

![Loss](results/training_loss.png)

### Model Comparison

![Comparison](results/model_comparison.png)

---

## Key Learnings

* Variational quantum circuits can be trained similarly to neural networks
* Feature encoding plays a critical role in QML performance
* Small quantum models have limited expressiveness compared to classical models
* Hybrid quantum-classical workflows are essential in current QML applications


## How to Run

```bash
git clone https://github.com/Devarshii/qml-moons-classifier.git
cd qml-moons-classifier
pip install -r requirements.txt
jupyter notebook
```

## Conclusion

This project demonstrates a complete QML workflow, from data preprocessing to model evaluation. While the quantum model is a simple example, it provides insight into how quantum machine learning models are built and trained in practice.


## Future Improvements

* Increase circuit depth and complexity
* Experiment with different datasets
* Implement using Qiskit
* Explore quantum kernel methods

