
---
title: "A Systematization of the Wagner Framework: Graph Theory Conjectures and Reinforcement Learning"
output: html_document
---

# Overview

The `main.py` file provides a graphical user interface (GUI) using `Tkinter` which enables the user to select a conjecture, choose a game type and the number of nodes, and interact with the game by performing actions. The GUI visualizes the current state of the game denoted by a graph $G$ along with the function $f(G)$ which we intend to maximize.

The custom game environments are implemented in separate files using the `Gym` framework, which allows them to take advantage of all extensions and tools available for Gym environments:

-   `linear_environment.py`: implementation of the Linear game.
-   `local_environment.py`: implementation of the Local game.
-   `global_environment.py`: implementation of the Global game.
-   `flip_environment.py`: implementation of the Flip game.

# Installation

To run this project, you need to have Python installed on your system along with the necessary libraries. Follow the steps below to set up the environment:

1. Install Python (version 3.6 or higher).
2. Install the required libraries using pip:

```sh
pip install tkinter matplotlib networkx numpy
```

3. Ensure the custom game environment modules (`linear_environment`, `local_environment`, `global_environment`, `flip_environment`) are available in your Python path.

# Usage

To start the GUI, run the main Python script:

```sh
python main.py
```

Follow the instructions on the screen to interact with the application.

## Step-by-Step Guide

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

## Example

Here is a simple example of how to run the application:

```sh
python main.py
```

1. Select *Wagner* as the conjecture.
2. Choose *Flip* as the game type, enter `5` for the number of nodes and `Sparse` for a sparse reward.
3. In the game window, enter actions and observe the updates to the graph state and relative rewards until a terminal state is reached.

# Code Structure

- `select_value_fun()`: Function to prompt the user to select a value function.
- `select_game(selected_value_fun)`: Function to prompt the user to select a game type, number of nodes and type of reward.
- `visualize_state(canvas, figure, graph, current)`: Function to visualize the current state of the game.
- `value_fun_wagner(graph, normalize)`: Implementation of the Wagner reward.
- `value_fun_brouwer(graph, normalize)`: Implementation of the Brouwer reward.
- `main_game(game_name, number_of_nodes, value_fun)`: Main function to run the selected game.

