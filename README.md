# A biological model of nonlinear dimensionality reduction

## Overview
These codes are accompanying the manuscript:

Kensuke Yoshida and Taro Toyoizumi,  

"A biological model of nonlinear dimensionality reduction"

Preprint: https://www.biorxiv.org/content/10.1101/2024.03.13.584757v1

## Requirements
Simulations for the paper above were conducted in the following setup:
- python 3.9.16, numpy 1.24.3, matplotlib 3.7.1, scipy 1.10.1, scikit-learn 1.2.2, minisom 2.3.3

## Usage
For basic usage, please refer to the following Jupyter notebook in the 'code' folder:
- 'biondr_demo.ipynb'

Details about the model can be found in 'model.py'. Other scripts are used for generating the figures included in the paper.

The 'data' folder contains the valence index data from [1], provided by the author of [1].  
Before running scripts related to Drosophila data analysis, please ensure that the following files are placed in the 'data' folder:
- The file containing the PN activities and the valence index from [1], named `odordata_Badel2016.csv`.
- The file containing the olfactory receptor responses from [2], named `odorspaceHallem2006.csv`.

These data are available at [1, 2].

[1] L. Badel, K. Ohta, Y. Tsuchimoto, H. Kazama, Decoding of Context-Dependent Olfactory Behavior in Drosophila. Neuron 91 (1), 155–167 (2016).

[2] E. A. Hallem, J. R. Carlson, Coding of odors by a receptor repertoire. Cell 125 (1), 143–160 (2006).

## License
This project is licensed under the MIT License (see LICENSE.txt for details).
