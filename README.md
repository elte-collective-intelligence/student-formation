# Formation Project

This project simulates multi-agent reinforcement learning using the PettingZoo framework. It includes several key scripts to define, train, and run the environment.

## Project Structure
- `src/`: Contains all source code files.
- `models/`: Pre-trained model files.
- `logs/`: Logs and TensorBoard files.
- `venv/`: Virtual environment folder (should not be shared).
- `data/`: Placeholder for input data.
- `outputs/`: Contains simulation results and other output files.
- `notebooks/`: Jupyter notebooks for analysis.

## File Descriptions

### 1. environment.py
Defines the simulation environment for multi-agent reinforcement learning using PettingZoo. Key features include;

- World setup, agent initialization, reward function, and rendering.
- Logic for resetting the environment and evaluating agent behaviors.

### 2. formation.py
Simulates agent interactions within the environment. Key features include;

- Loads pre-trained models for circle and triangle tasks.
- Executes agent actions and updates the environment state.

### 3. trainer.py
Trains reinforcement learning models using the PettingZoo environment and stable-baselines3:
- Implements the PPO algorithm for multi-agent learning.
- Logs training data and saves models iteratively.


## Running the Code

### 1. Extract Models
Unzip circle_model.zip and mountains_model.zip;

```bash
unzip models/circle_model.zip -d models/circle
unzip models/mountains_model.zip -d models/mountains
```

### 2. Install Dependencies

```bash
pip install -r dependencies.txt
```
The dependencies.txt file contains the following packages;
- numpy: Numerical operations (used for calculations like distance and positions).
- pygame: Visualization and rendering for the environment.
- shapely: Geometric operations, such as defining shapes and calculating intersections.
- pettingzoo: Framework for multi-agent reinforcement learning.
- stable-baselines3[extra]: Reinforcement learning algorithms with additional dependencies for environments like PettingZoo.
- supersuit: Wrappers for processing observations and actions in PettingZoo environments.
- imageio: Creating GIFs from simulation frames.
- matplotlib: Visualization of training data or simulation results.

### 3. Train the models

```bash
python trainer.py
```

### 4. Run the simulation

```bash
python formation.py
```


## Converting the Simulation to GIF

### 1. Install FFmpeg;
```
sudo apt install ffmpeg
```

### 2. Modify Code for GIF Conversion;
Code snippet to add in "formation.py"

``` python
import imageio

frames = []  # List to store frames
for step in range(simulation_steps):
    frame = env.render(mode='rgb_array')  # Get frame from environment
    frames.append(frame)

# Save as GIF
imageio.mimsave('simulation.gif', frames, fps=30)
```

### 3. Generate the GIF;
```bash
python formation.py
```
A "simulation.gif" file should be generated in the directory.


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.


## License

[MIT](https://choosealicense.com/licenses/mit/)
