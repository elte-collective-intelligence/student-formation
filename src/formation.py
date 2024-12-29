import os
import json
import numpy as np
from PIL import Image, ImageDraw
import imageio
from stable_baselines3 import PPO
from src.environment import env

# Define the root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load configuration file
config_path = os.path.join(ROOT_DIR, 'config.json')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Extract configuration details
num_good_guys = config["num_good_guys"]
num_bad_guys = config["num_bad_guys"]
shape_settings = config["shape_settings"]

# Initialize the environment
frame_width, frame_height = 800, 600  # Increased resolution
total_agents = num_good_guys + num_bad_guys
raw_env = env(num_agents=total_agents, max_cycles=500, render_mode='rgb_array')
raw_env.reset()

# Define custom objects to replace incompatible parameters
custom_objects = {
    "clip_range": lambda _: 0.2,
    "lr_schedule": lambda _: 0.001
}

# Load models
circle_model_path = os.path.join(ROOT_DIR, 'models', 'circle_model.zip')
triangle_model_path = os.path.join(ROOT_DIR, 'models', 'mountains_model.zip')

if not os.path.exists(circle_model_path) or not os.path.exists(triangle_model_path):
    raise FileNotFoundError("Required model files are missing in the 'models' directory.")

model_good = PPO.load(circle_model_path, custom_objects=custom_objects)
model_bad = PPO.load(triangle_model_path, custom_objects=custom_objects)

# Function to create polygons
def create_polygon(image_size, num_sides, radius, center, color):
    """
    Create a fixed polygon in a 2D space.
    :param image_size: Tuple (width, height) of the image.
    :param num_sides: Number of sides of the polygon.
    :param radius: Radius of the polygon.
    :param center: Tuple (x, y) for the center of the polygon.
    :param color: Fill color of the polygon in RGBA format.
    :return: PIL Image with the polygon drawn.
    """
    image = Image.new("RGBA", image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    points = [
        (
            center[0] + radius * np.cos(2 * np.pi * i / num_sides),
            center[1] + radius * np.sin(2 * np.pi * i / num_sides),
        )
        for i in range(num_sides)
    ]

    draw.polygon(points, fill=color)
    return image

# Initialize agent positions randomly within the frame
agent_positions = {
    f"agent_{i}": (np.random.randint(0, frame_width), np.random.randint(0, frame_height))
    for i in range(total_agents)
}

# Create a list to store frames
frames = []

# Define fixed polygon and mountain details
fixed_polygon_center = (400, 150)  # Center position of the polygon
fixed_polygon_color = (0, 255, 0, 255)  # Green color
mountain_color = (0, 0, 255, 255)  # Blue color

# Run the simulation
step = 0
for agent in raw_env.agent_iter():
    observation, reward, termination, truncation, info = raw_env.last()

    # Extract numeric part of the agent identifier
    agent_id = int(agent.split("_")[1])

    if agent_id < num_good_guys:
        # Good guys (blue circle)
        action = model_good.predict(observation, deterministic=True)[0]
        agent_is_good = True
        shape_config = shape_settings[1]  # Square for good guys
    else:
        # Bad guys (no circle)
        action = model_bad.predict(observation, deterministic=True)[0]
        agent_is_good = False
        shape_config = shape_settings[0]  # Triangle for bad guys

    if termination or truncation:
        action = None

    # Update agent positions and remove stuck agents
    for agent_id, pos in list(agent_positions.items()):
        x, y = pos
        if x <= 10 and y >= frame_height - 10:  # Bottom-left stuck condition
            del agent_positions[agent_id]  # Remove the agent
        else:
            x = np.clip(x, 0, frame_width)
            y = np.clip(y, 0, frame_height)
            agent_positions[agent_id] = (x, y)

    # Step the environment
    raw_env.step(action)
    step += 1

    # Capture frames
    if step % 10 == 0:
        try:
            frame = raw_env.render()
            frame_image = Image.fromarray(frame).resize((frame_width, frame_height))

            # Draw the fixed polygon
            fixed_polygon = create_polygon(
                image_size=(frame_width, frame_height),
                num_sides=6,
                radius=50,
                center=(420, 180), 
                color=fixed_polygon_color,
            )
            frame_image = Image.alpha_composite(frame_image.convert("RGBA"), fixed_polygon)

            # Draw the mountain in blue (moved closer to the center)
            mountain_overlay = create_polygon(
                image_size=(frame_width, frame_height),
                num_sides=5,
                radius=80,
                center=(220, 380), 
                color=mountain_color,
            )
            frame_image = Image.alpha_composite(frame_image, mountain_overlay)

            # Convert to RGBA for further processing
            frames.append(frame_image)
        except Exception as e:
            print(f"Error capturing frame: {e}")

raw_env.close()

# Save the frames as a GIF
output_path = os.path.join(ROOT_DIR, 'outputs', 'simulation.gif')
if frames:
    imageio.mimsave(output_path, frames, fps=30)
    print(f"Simulation saved as GIF at: {output_path}")
else:
    print("No frames were captured during the simulation.")