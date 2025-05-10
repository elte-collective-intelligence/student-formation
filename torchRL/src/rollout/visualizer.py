import imageio
import matplotlib.pyplot as plt
import os

def generate_gif(frames, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimsave(filename, frames, duration=0.1)

def record_episode(env, policy, gif_path="outputs/gifs/formation.gif"):
    td = env.reset()
    done = torch.tensor([False])
    frames = []

    while not done.item():
        fig, ax = plt.subplots()
        positions = td["obs"][..., :2].detach().numpy()
        ax.scatter(positions[:, 0], positions[:, 1])
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title("Agent Positions")
        fig.canvas.draw()

        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

        action_td = policy(td)
        td = env.step(action_td)
        done = td["done"]

    generate_gif(frames, gif_path)
