
---
A Systematization of the Wagner Framework: Graph Theory Conjectures and Reinforcement Learning
---
This repository contains the supplementary materials as described in the paper [here](https://arxiv.org/abs/2406.12667).

# Environments
## Overview

The `main.py` file provides a graphical user interface (GUI) using `Tkinter` which enables the user to select a conjecture, choose a game type and the number of nodes, and interact with the game by performing actions. The GUI visualizes the current state of the game denoted by a graph $G$ along with the function $f(G)$ which we intend to maximize.

The custom game environments are implemented in separate files using the `Gym` framework, which allows them to take advantage of all extensions and tools available for Gym environments:

-   `linear_environment.py`: implementation of the Linear game.
-   `local_environment.py`: implementation of the Local game.
-   `global_environment.py`: implementation of the Global game.
-   `flip_environment.py`: implementation of the Flip game.

## Installation

To run this project, you need to have Python installed on your system along with the necessary libraries. Follow the steps below to set up the environment:

1. Install Python (version 3.6 or higher).
2. Install the required libraries using pip:

```sh
pip install tkinter matplotlib networkx numpy
```

3. Ensure the custom game environment modules (`linear_environment`, `local_environment`, `global_environment`, `flip_environment`) are available in your Python path.

## Usage

To start the GUI, run the main Python script:

```sh
python main.py
```

Follow the instructions on the screen to interact with the application.

### Step-by-Step Guide

1. **Select a Conjecture**: 
    - Upon running the script, a window will prompt you to select a conjecture. You can choose between *Wagner* and *Brouwer*.
    
2. **Select a Game**:
    - After selecting the conjecture, another window will appear asking you to choose a game type, specify the number of nodes in the graph and the type of reward. Available game types are *Linear*, *Local*, *Global*, and *Flip*, while the reward can be chosen between *Sparse* and *Incremental*.
    
3. **Interact with the Game**:
    - Once the game starts, you can enter actions in the provided entry box and press *Execute* to perform the action. The state of the graph will be updated and displayed on the screen.
    - **Edge Colors**:
      - **Red Line**: Represents the current edge.
        - **Solid Red Line**: Indicates that the current edge is present.
        - **Dashed Red Line**: Indicates that the current edge is not present.
      - **Grey Line**: Represents other edges in the graph.
    - Click on *Exit Game* to quit the application at any time.

### Example

Here is a simple example of how to run the application:

```sh
python main.py
```

1. Select *Wagner* as the conjecture.
2. Choose *Flip* as the game type, enter `5` for the number of nodes and `Sparse` for a sparse reward.
3. In the game window, enter actions and observe the updates to the graph state and relative rewards until a terminal state is reached.

## Code Structure

- `select_value_fun()`: Function to prompt the user to select a value function.
- `select_game(selected_value_fun)`: Function to prompt the user to select a game type, number of nodes and type of reward.
- `visualize_state(canvas, figure, graph, current)`: Function to visualize the current state of the game.
- `value_fun_wagner(graph, normalize)`: Implementation of the Wagner reward.
- `value_fun_brouwer(graph, normalize)`: Implementation of the Brouwer reward.
- `main_game(game_name, number_of_nodes, value_fun)`: Main function to run the selected game.

# Dataset

## Files Description
The dataset was originally designed for experiments on Brouwer's conjecture. It contains 11983 graphs with 11 vertices, each one labelled with its laplacian spectra. All dataset's information are divided into three files:

- <b>*n11_graphs.g6*</b>: contains the g6 encoding of the graphs. The .g6 format is a compact text-based encoding. It is well-supported and can be easily read by python NetworkX method *read_graph6*.
- <b>*n11_laplacian_spectra.txt*</b>: contains the labels. Each line includes 11 laplacian eigenvalues, reported in descending order and separated by spaces. The *i-th* row of this files contains the eigenvalues of the *i-th* graph in *n11_graphs.g6*.
- <b>*weisfeiler_leman_results.txt*</b>: in this files each row contains a numeric pair, with numbers separated by space. A number *i* represents the graph in the *i-th* row of *n11_graphs.g6*. The reported pairs are the ones that succeeded in 1-dimensional Weifeiler-Leman test, which implies that the two graphs involved could be isomorphic. The total number of such pairs is 1124695.

## Dataset generation
The dataset integrates 11-vertices graphs downloaded from *House of Graphs* online database with random graphs generated via NetworkX implementations of Erdős–Rényi (ER), Watts-Strogatz (WS) and Barabási–Albert (BA) models, in quantities:

- 1010 graphs drawn from ER models, considering the probability $p$ varying in $[0,1]$ with step $0.01$. We draw 10 graphs from each choice of $p$.
- 540 graphs drawn from WS models, varying the mean degree $k$ in $\{4,6,8\}$, and rewriting edges' probability $\beta$ in $[0.1,0.9]$ with step $0.1$. This gave us 27 different models and 20 graphs were drawn from each.
- 10162 graphs obtained from BA models, with parameter *m* varying in $\{2,...,9\}$. BA algorithm builds a graph starting from an initial configuration on $m_0>m$ nodes. We took House of Graphs' samples with $3 \leq n \leq 10$ and used each $G$ in this batch to start a BA generation with $m<|V(G)|$.
- 271 graphs downloaded from *The House of Graphs*. The database collects non-isomorphic graphs only.

Each of this batches was generated and labelled with its own routine. The 4 groups were then randomly mixed in one single file, to maintain the variety of considered graphs when dividing into minibatches.
