# T-PDG
Transformer-based Powered Descent Guidance (T-PDG), a scalable algorithm for reducing the computational complexity of the direct optimization formulation of the spacecraft powered descent guidance problem.

![T-PDG Trajectories](https://github.com/jujubriden/T-PDG/blob/main/src/Results/T-PDG%20Trajectory.gif)

The lossless convexification (LCvx) algorithm, which was used for problem training and test data, was adapted from the [SCP Toolbox](https://github.com/UW-ACL/SCPToolbox.jl)[^1].

## To Run:

1. Make sure [SCP Toolbox](https://github.com/UW-ACL/SCPToolbox.jl) is also installed.

2. Ensure PyCall is installed with the correct Python path and LaTeX is downloaded.

3. In Julia run include("Tests/run_tests.jl") inside of the T-PDG folder.

## To Design New Models, Train & Test, or Visualize Models with t-SNE:

1. Open Tests/NN_Train_and_Test.ipynb and navigate to the most relevant section for your task.

## Included Folders and Files
* T-PDG
    * src - Contains required files for running the algorithm
        * Data - .pkl files including mean and standard deviations for the datasets are stored here, as well as standardized training and testing data
        * Models - Trained transformer models are stored here
        * Results - Result figures and datasets are saved here
        * Sampling - .csv files sampled from LCvx with tight constraints and optimal final times are stored here
        * definition.jl - LCvx optimization problem created, constraints are added, and the optimization problem is solved
        * parameters.jl - Constructors for setting up Rocket and Solution structures
        * T-PDG.jl - Creates a package from the src files
        * tests.jl - Tests the T-PDG algorithm and compares runtime and feasibility with LCvx
    * Tests - Contains files for running the guidance algorithm and plots
        * NN_Train_and_Test.ipynb - Preprocess data, train and test transformer neural networks, and visualize embeddings using t-SNE
        * plots.jl - Contains all plotting functions
        * run_tests.jl - Run T-PDG using 
        
                include("Tests/run_tests.jl")

[^1]: Danylo Malyuta, Taylor P. Reynolds, Michael Szmuk, Thomas Lew, Riccardo Bonalli, Marco Pavone, Behçet Açıkmeşe. "Convex Optimization for Trajectory Generation: A Tutorial on Generating Dynamically Feasible Trajectories Reliably and Efficiently". *IEEE Control Systems*, 42(5), pp. 40-113, 2022. [DOI: 10.1109/mcs.2022.3187542](https://doi.org/10.1109/mcs.2022.3187542). Free preprint available at [https://arxiv.org/abs/2106.09125](https://arxiv.org/abs/2106.09125)

