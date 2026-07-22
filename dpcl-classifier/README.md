# Generalized Convergence Analysis of Tsetlin Machines: A Probabilistic Approach to Concept Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.13.0-brightgreen.svg)](https://www.tensorflow.org/)

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

Tsetlin Machines (TMs) have garnered increasing interest for their ability to learn concepts via propositional formulas and their proven efficiency across various application domains. Despite this, the convergence proof for the TMs, particularly for the AND operator (_conjunction_ of literals), in the generalized case (inputs greater than two bits) remains an open problem. This paper aims to fill this gap by presenting a comprehensive convergence analysis of Tsetlin automaton-based Machine Learning algorithms. We introduce a novel framework, referred to as Probabilistic Concept Learning (PCL), which simplifies the TM structure while incorporating dedicated feedback mechanisms and dedicated inclusion/exclusion probabilities for literals. Given $n$ features, PCL aims to learn a set of conjunction clauses $C_i$ each associated with a distinct inclusion probability $p_i$. Most importantly, we establish a theoretical proof confirming that, for any clause $C_k$, PCL converges to a conjunction of literals when $0.5 < p_k <1 $.
This result serves as a stepping stone for future research on the convergence properties of Tsetlin automaton-based learning algorithms. Our findings not only contribute to the theoretical understanding of Tsetlin Machines but also have implications for their practical application, potentially leading to more robust and interpretable machine learning models.

## Installation

### Prerequisites

- Python 3.11 or higher
- Poetry package manager

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/cair/dpcl-classifier.git
    ```

2. Navigate to the project directory:

   ```bash
   cd dpcl-classifier
   ```

3. Install the dependencies using Poetry:

   ```bash
   poetry install
   ```

## Usage

You can run the main script as follows:
    ```bash
    python classifier_numba.py
    ```

## License 

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
