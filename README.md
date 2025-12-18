# Formation: Multi-Agent Reinforcement Learning (TorchRL)

[![CI](https://github.com/elte-collective-intelligence/student-formation/actions/workflows/ci.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-formation/actions/workflows/ci.yml)
[![Docker](https://github.com/elte-collective-intelligence/student-formation/actions/workflows/docker.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-formation/actions/workflows/docker.yml)
[![codecov](https://codecov.io/gh/elte-collective-intelligence/student-formation/branch/main/graph/badge.svg)](https://codecov.io/gh/elte-collective-intelligence/student-formation)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-blue.svg)](LICENSE)

## About the project

This project implements a multi-agent reinforcement learning (MARL) system using TorchRL to train agents to form specific geometric shapes. The agents learn cooperative behavior through PPO (Proximal Policy Optimization) to achieve formation control.

### Key Features

- **Multiple Shape Support**: Circle, Polygon, and Star formations with dynamic reconfiguration
- **Assignment Strategies**: Hungarian and Greedy assignment strategies for optimal agent-target matching
- **Reward Functions**: Support for both SDF-based (shape boundary) and assignment-based (target position) rewards
- **Multi-Shape Scenes**: Agents can be assigned to multiple different shapes simultaneously
- **Visualization**: Real-time rendering and GIF generation of trained policies
- **Testing & CI/CD**: Comprehensive test suite with automated pipelines

### How It Works

1. **Environment**: Agents are placed in an arena and receive observations of their relative positions
2. **Target Formation**: A geometric shape defines the desired formation
3. **Assignment**: Agents are assigned to specific target positions using Hungarian or Greedy algorithms
4. **Training**: PPO trains agents to move toward their assigned positions while respecting arena boundaries
5. **Evaluation**: Trained models can be visualized and evaluated on formation accuracy

### Technologies

- **TorchRL**: Multi-agent reinforcement learning framework
- **PyTorch**: Deep learning backend
- **Hydra**: Configuration management
- **Weights & Biases**: Experiment tracking and visualization

## Setup

1. Make sure you have python version 3.11 or at least 3.10  
Check by running `python --version`. If you have older version please update.

2. Create virtual environment and activate it

```shell
python -m venv .venv && source .venv/bin/activate
```

3. Upgrade pip

```shell
python -m pip install --upgrade pip
```

4. Install runtime dependencies

```shell
pip install -r requirements.txt
```

## Usage

### Training

To train agents on a formation task, use:

```shell
python main.py
```

The default configuration trains agents to form a circle. Output including training metrics and model checkpoints are logged to W&B.

### Configuration

Training behavior is controlled through YAML config files in the `configs/` directory:

- **`configs/base/main_setup.yaml`**: Global settings (device, seed, project name)
- **`configs/algo/ppo.yaml`**: PPO algorithm hyperparameters (learning rate, epochs, clip epsilon)
- **`configs/env/formation.yaml`**: Environment settings (num_agents, arena_size, shape_type)
- **`configs/experiment/default_exp.yaml`**: Experiment configuration (combines all above)

#### Defining Shapes

Shapes are defined in `configs/env/formation.yaml`. Each shape type has specific parameters:

**Circle Formation**

```yaml
shape_type: circle
circle:
  center: [0.0, 0.0]    # Center coordinates [x, y]
  radius: 2.0           # Circle radius
```

**Polygon Formation**

```yaml
shape_type: polygon
polygon:
  vertices: [           # List of [x, y] vertices
    [-2.0, -2.0],
    [2.0, -2.0],
    [2.0, 2.0],
    [-2.0, 2.0]
  ]
```
Supports both convex and non-convex polygons. Agents are distributed evenly along the perimeter.

**Star Formation**

```yaml
shape_type: star
star:
  center: [0.0, 0.0]    # Center coordinates
  r1: 1.0               # Inner radius
  r2: 2.0               # Outer radius
  n_points: 5           # Number of star points
```

#### Multi-Shape Scenes with Reconfiguration

For complex scenarios with multiple shapes, use the `multishape` type:

```yaml
shape_type: multishape

multishape:
  shapes:
    - type: circle
      center: [-3.0, 0.0]
      radius: 1.5
      agent_count: 5     # Agents assigned to this shape
    
    - type: polygon
      vertices: [
          [2.0, -2.0],
          [4.0, -2.0],
          [4.0, 2.0],
          [2.0, 2.0]
        ]
      agent_count: 5     # Remaining agents assigned here

# Dynamic reconfiguration (switch formations mid-episode)
reconfig_step: 200      # When should the reconfiguration happen
reconfig_shape:
  shape_type: multishape    # Shape defined to switch to
  multishape:
    - type: polygon
      vertices: [[-4.0, 0.0], [-2.0, 0.0], [-2.0, -2.0], [-4.0, -2.0]] 
      agent_count: 5
    - type: circle
      center: [3.0, 0.0]
      radius: 1.5
      agent_count: 5
```

#### Assignment Strategies

Choose how agents are assigned to target positions:

```yaml
# Hungarian algorithm (optimal but slower)
assignment_method: "hungarian"

# Greedy algorithm (faster, near-optimal)
assignment_method: "greedy"
```

#### Example Configurations

**Circle with Hungarian Assignment**

```yaml
shape_type: "circle"
circle:
  center: [0.0, 0.0]
  radius: 2.0
assignment_method: "hungarian"
num_agents: 10
```

**Multi-Shape with Reconfiguration**

```yaml
shape_type: multishape
num_agents: 20

multishape:
  shapes:
    - type: circle
      center: [-2.0, 0.0]
      radius: 1.5
      agent_count: 10
    - type: star
      center: [2.0, 0.0]
      r1: 0.8
      r2: 1.8
      n_points: 5
      agent_count: 10

reconfig_shape:
  shape_type: multishape
  multishape:
    - type: polygon
      vertices: [[0, -2], [2, 0], [0, 2], [-2, 0]]
      agent_count: 10
    - type: circle
      center: [0.0, 0.0]
      radius: 2.0
      agent_count: 10
```

### Visualization

After training, visualize the learned policy using:

```shell
python visualize.py
```

This script:

- Loads the most recent trained model from W&B
- Runs the policy in the environment for several episodes
- Renders real-time visualization of agents forming the target shape
- Generates a GIF of the formation process
- Displays formation accuracy and episode metrics

## Running tests

```shell
python -m unittest discover -s test
```

## Work distribution

### Márk Baricz

- SDF interface, including support for three shapes (circle, polygon, star)
- Redesign of the observations and rewards with SDF terms
- Render support for the new shapes
- Fixing the visualizer script and GIF generation (visualizing target shapes and positions, loading trained model)
- Implementation of the Hungarian and Greedy assignment strategies
- Support for multi-shape scenes
- Support for dynamic reconfiguration mid-episode (even with multi-shape scenes)
- Fixing the tests and CI/CD pipelines