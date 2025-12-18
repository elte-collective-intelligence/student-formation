import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.data.tensor_specs import (
    UnboundedContinuousTensorSpec as Unbounded,
    CompositeSpec as Composite,
    BoundedTensorSpec as Bounded,
    DiscreteTensorSpec,
)
from scipy.optimize import linear_sum_assignment
import numpy as np
import pygame  # For rendering
import pygame.gfxdraw

from src.envs.shapes import (
    Circle,
    MultiShape,
    Polygon,
    make_star_vertices,
)  # For anti-aliased shapes

# --- Helper for printing warnings only once ---
_printed_warnings = set()


def print_warning_once(message):
    if message not in _printed_warnings:
        print(f"WARNING: {message}")
        _printed_warnings.add(message)


class FormationEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, cfg, device="cpu"):
        super().__init__(device=device, batch_size=torch.Size([cfg.env.num_agents]))

        self.cfg = cfg
        self.num_agents = cfg.env.num_agents
        self.arena_size = cfg.env.arena_size
        self.max_steps = cfg.env.max_steps

        self.agent_size_world_units = cfg.env.get(
            "agent_size", 0.05
        )  # Diameter in world units
        self.agent_accel = cfg.env.get("agent_accel", 3.0)
        self.agent_max_speed = cfg.env.get("agent_max_speed", 0.3)
        self.dt = 0.1

        self.agent_positions = torch.zeros(self.num_agents, 2, device=self.device)
        self.agent_velocities = torch.zeros(self.num_agents, 2, device=self.device)
        self.current_step = 0

        # Reconfiguration step
        self.reconfig_step = cfg.env.get("reconfig_step", None)
        self.has_reconfigured = False

        # Shape configuration
        self.shape_type = cfg.env.get("shape_type", "circle")
        self.target_shape = self.__create_shape(self.shape_type, cfg.env)

        # Assignment strategy
        self.assignment_method = cfg.env.get("assignment_method", "greedy")
        self.shape_boundary_points = self.target_shape.get_target_points(
            self.num_agents
        )
        self.assigned_target_positions = torch.zeros(
            self.num_agents, 2, device=self.device
        )

        self._make_specs()
        self.actor_obs_keys = cfg.env.obs_keys_for_actor

        # Rendering attributes
        self.screen = None
        self.clock = None
        self.render_scale = 100  # Pixels per world unit (approx)
        self.screen_width = 800
        self.screen_height = 600
        self.render_initialized = False
        self.window_closed_by_user = False

    def __create_shape(self, shape_type, cfg):
        if shape_type == "circle":
            # Look inside the source for 'circle' params
            c_cfg = cfg.circle
            return Circle(c_cfg.center, c_cfg.radius, self.device)

        elif shape_type == "polygon":
            # Look inside source for 'polygon' params
            verts = torch.tensor(
                cfg.polygon.vertices,
                device=self.device,
                dtype=torch.float32,
            )
            return Polygon(verts, device=self.device)

        elif shape_type == "star":
            s_cfg = cfg.star
            verts = make_star_vertices(s_cfg.center, s_cfg.r1, s_cfg.r2, s_cfg.n_points)
            return Polygon(verts, device=self.device)

        elif shape_type == "multishape":
            sub_shapes = []
            counts = []
            # Look inside source for 'multishape' list
            for s_cfg in cfg.multishape:
                t = s_cfg.type
                counts.append(s_cfg.agent_count)

                if t == "circle":
                    s = Circle(s_cfg.center, s_cfg.radius, self.device)
                elif t == "polygon":
                    v = torch.tensor(
                        s_cfg.vertices, device=self.device, dtype=torch.float32
                    )
                    s = Polygon(v, device=self.device)
                elif t == "star":
                    # Assume star params are inline in the list item
                    v = make_star_vertices(
                        s_cfg.center, s_cfg.r1, s_cfg.r2, s_cfg.n_points
                    )
                    s = Polygon(
                        torch.tensor(v, device=self.device, dtype=torch.float32),
                        self.device,
                    )
                sub_shapes.append(s)

            return MultiShape(sub_shapes, counts, self.device)

        else:
            raise ValueError(f"Unsupported shape_type: {shape_type}")

    def _trigger_reconfiguration(self):
        if "reconfig_shape" in self.cfg.env:
            # 1. Get the new config block
            new_cfg = self.cfg.env.reconfig_shape
            new_type = new_cfg.shape_type

            # 2. Build shape using that block
            self.shape_type = new_type
            self.target_shape = self.__create_shape(new_type, new_cfg)

            # 3. Update Targets
            self.shape_boundary_points = self.target_shape.get_target_points(
                self.num_agents
            )
            self.__update_assignments()

    def __update_assignments(self):
        dists = torch.cdist(self.agent_positions, self.shape_boundary_points)
        if self.assignment_method == "hungarian":
            cost_matrix = dists.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assignments = sorted(zip(row_ind, col_ind), key=lambda x: x[0])
            sorted_col_ind = [x[1] for x in assignments]

            assigned_indices = torch.tensor(sorted_col_ind, device=self.device)
            self.assigned_target_positions = self.shape_boundary_points[
                assigned_indices
            ]
        elif self.assignment_method == "greedy":
            vals, indices = torch.min(dists, dim=1)
            self.assigned_target_positions = self.shape_boundary_points[indices]

    def _make_specs(self) -> None:
        obs_dim_per_agent = 5  # sdf(1) + target_vec(2) + closest_agent_vec(2)
        self.observation_spec = Composite(
            {
                "observation": Unbounded(
                    shape=(self.num_agents, obs_dim_per_agent), device=self.device
                )
            },
            shape=torch.Size([self.num_agents]),
        )
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(self.num_agents, 2),
            device=self.device,
            dtype=torch.float32,
        )
        self.reward_spec_unbatched = Unbounded(shape=(1,), device=self.device)
        self.done_spec_unbatched = DiscreteTensorSpec(
            n=2, shape=torch.Size([1]), dtype=torch.bool, device=self.device
        )

    def _get_observations(self) -> torch.Tensor:
        sdf = self.target_shape.signed_distance(self.agent_positions).unsqueeze(1)

        dist_matrix = torch.cdist(self.agent_positions, self.agent_positions)
        dist_matrix.fill_diagonal_(float("inf"))
        if self.num_agents > 1:
            _, closest_indices = torch.min(dist_matrix, dim=1)
            closest_agent_positions = self.agent_positions[closest_indices]
            vec_to_closest = closest_agent_positions - self.agent_positions
        else:
            vec_to_closest = torch.zeros_like(self.agent_positions)

        target_vec = self.assigned_target_positions - self.agent_positions

        observations = torch.cat(
            [sdf, target_vec, vec_to_closest], dim=1
        )  # [N, 1+2+2=5]

        # Safety check
        observations = torch.nan_to_num(
            observations, nan=0.0, posinf=100.0, neginf=-100.0
        )
        observations = torch.clamp(observations, -100.0, 100.0)

        return observations

    def _reset(self, tensordict: TensorDictBase = None) -> TensorDictBase:
        self.current_step = 0

        self.shape_type = self.cfg.env.get("shape_type", "circle")
        self.target_shape = self.__create_shape(self.shape_type, self.cfg.env)
        self.shape_boundary_points = self.target_shape.get_target_points(
            self.num_agents
        )

        min_val, max_val = -self.arena_size / 2, self.arena_size / 2
        self.agent_positions = (
            torch.rand(self.num_agents, 2, device=self.device) * (max_val - min_val)
        ) + min_val
        self.agent_velocities = torch.zeros(self.num_agents, 2, device=self.device)
        self.__update_assignments()
        current_observations = self._get_observations()
        done_val = torch.zeros(
            (self.num_agents, 1), dtype=torch.bool, device=self.device
        )
        return TensorDict(
            {
                "observation": current_observations,
                "done": done_val,
            },
            batch_size=torch.Size([self.num_agents]),
            device=self.device,
        )

    def _calc_rewards(self) -> torch.Tensor:
        rewards = torch.zeros(self.num_agents, 1, device=self.device)

        sdf = self.target_shape.signed_distance(self.agent_positions).unsqueeze(1)
        formation_accuracy_reward = torch.exp(-5.0 * sdf**2)
        rewards += formation_accuracy_reward

        dist_to_assigned = torch.norm(
            self.assigned_target_positions - self.agent_positions, dim=1, keepdim=True
        )
        assignment_accuracy_reward = torch.exp(-2.0 * dist_to_assigned**2)
        rewards += assignment_accuracy_reward

        # Boundary penalty (from your MPE code, applied universally)
        # This assumes arena is roughly [-1, 1] if boundary is 0.9
        # We should use self.arena_size for boundary checks.
        normalized_pos = self.agent_positions / (
            self.arena_size / 2.0
        )  # Normalize to roughly [-1, 1]
        abs_norm_pos = torch.abs(normalized_pos)

        penalty = torch.zeros_like(abs_norm_pos)  # [N, 2]
        # Thresholds for penalty based on normalized position
        bound_thresh_soft = 0.95  # Start penalty slightly inside the arena edge
        bound_thresh_hard = 1.0  # Max penalty at/beyond arena edge

        cond2 = (abs_norm_pos >= bound_thresh_soft) & (abs_norm_pos < bound_thresh_hard)
        penalty[cond2] = (
            abs_norm_pos[cond2] - bound_thresh_soft
        ) * 20  # Scaled penalty

        cond3 = abs_norm_pos >= bound_thresh_hard
        # Stronger penalty if outside the hard boundary
        penalty[cond3] = (
            torch.min(
                torch.exp(5 * (abs_norm_pos[cond3] - bound_thresh_hard)),
                torch.tensor(10.0, device=self.device),
            )
            + (abs_norm_pos[cond3] - bound_thresh_hard) * 20
        )  # Ensure it grows

        total_penalty_per_agent = torch.sum(penalty, dim=1, keepdim=True)  # [N, 1]
        rewards -= total_penalty_per_agent

        return rewards

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.current_step += 1

        # Check for reconfiguration
        if self.reconfig_step and self.current_step == self.reconfig_step:
            if not self.has_reconfigured:
                self._trigger_reconfiguration()
                self.has_reconfigured = True

        self.__update_assignments()
        actions = tensordict["action"]
        force = actions * self.agent_accel
        self.agent_velocities += force * self.dt
        speed = torch.norm(self.agent_velocities, dim=1, keepdim=True)
        too_fast_mask = speed > self.agent_max_speed
        safe_speed = speed.clone()
        safe_speed[speed == 0] = 1e-6
        mask_for_vel_update = too_fast_mask.squeeze(-1)
        if mask_for_vel_update.any():
            self.agent_velocities[mask_for_vel_update] = (
                self.agent_velocities[mask_for_vel_update]
                / safe_speed[mask_for_vel_update]
            ) * self.agent_max_speed
        self.agent_positions += self.agent_velocities * self.dt

        # Keep agents within defined arena_size (hard clamp)
        self.agent_positions = torch.clamp(
            self.agent_positions, -self.arena_size / 2, self.arena_size / 2
        )

        current_rewards = self._calc_rewards()
        is_episode_done = self.current_step >= self.max_steps
        done_val = torch.full(
            (self.num_agents, 1), is_episode_done, dtype=torch.bool, device=self.device
        )
        next_observations = self._get_observations()
        return TensorDict(
            {
                "observation": next_observations,
                "reward": current_rewards,
                "done": done_val,
            },
            batch_size=torch.Size([self.num_agents]),
            device=self.device,
        )

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("MARL Formation - TorchRL")
        self.clock = pygame.time.Clock()
        # Calculate render_scale based on arena_size fitting into screen dimensions
        # We want to map world coordinates from roughly -arena_size/2 to +arena_size/2
        # to screen coordinates 0 to screen_width/height.
        world_span = self.arena_size
        screen_span_w = self.screen_width * 0.9  # Use 90% of screen for margin
        screen_span_h = self.screen_height * 0.9
        self.render_scale = min(screen_span_w / world_span, screen_span_h / world_span)
        self.render_offset_x = self.screen_width / 2
        self.render_offset_y = self.screen_height / 2
        self.render_initialized = True
        self.window_closed_by_user = False

    def _to_screen_coords(self, world_pos_tensor):  # world_pos_tensor is [N, 2] or [2]
        # World origin (0,0) maps to screen center.
        # Pygame y is inverted.
        screen_pos = world_pos_tensor.clone()
        screen_pos[:, 1] *= -1  # Invert y-axis for Pygame
        screen_pos *= self.render_scale
        screen_pos += torch.tensor(
            [self.render_offset_x, self.render_offset_y], device=self.device
        )
        return screen_pos.cpu().numpy().astype(int)  # Return as int numpy array

    def render(self, mode="human"):
        if self.window_closed_by_user:  # If user closed window, don't try to render
            if mode == "rgb_array":  # Still need to return an array for GIF
                return np.zeros(
                    (self.screen_height, self.screen_width, 3), dtype=np.uint8
                )
            return None

        if not self.render_initialized and mode == "human":
            self._init_render()
        elif mode == "rgb_array" and self.screen is None:
            self._init_render()
            # pygame.display.iconify() # Optional: hide window if only for rgb_array

        if self.screen is None and mode == "human":  # Should have been initialized
            self._init_render()
        elif self.screen is None and mode == "rgb_array":
            self._init_render()  # Make sure screen is available for surfarray

        self.screen.fill((255, 255, 255))

        # --- Draw Target Shape ---
        def draw_one(s):
            if isinstance(s, Circle):
                c = self._to_screen_coords(s.center.unsqueeze(0))[0]
                r = int(s.radius * self.render_scale)
                pygame.gfxdraw.aacircle(self.screen, c[0], c[1], r, (200, 200, 200))
            elif isinstance(s, Polygon):
                v = self._to_screen_coords(s.vertices)
                pygame.draw.aalines(self.screen, (200, 200, 200), True, v.tolist())

        if isinstance(self.target_shape, MultiShape):
            for sub_s in self.target_shape.shapes:
                draw_one(sub_s)
        else:
            draw_one(self.target_shape)

        # --- Draw Agents ---
        agent_screen_pos = self._to_screen_coords(self.agent_positions)
        s_agent_radius = int(self.agent_size_world_units * self.render_scale / 2)
        s_agent_radius = max(s_agent_radius, 2)

        for i in range(self.num_agents):
            color = (70, 180, 70)
            pos = tuple(agent_screen_pos[i])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], s_agent_radius, color)
            pygame.gfxdraw.filled_circle(
                self.screen, pos[0], pos[1], s_agent_radius, color
            )
            # pygame.draw.circle(self.screen, (0,0,0), pos, s_agent_radius, 1) # Border

        if mode == "human":
            pygame.display.flip()
            if self.clock:
                self.clock.tick(self.metadata["render_fps"])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.window_closed_by_user = True
                    # self.render_initialized = False
                    # self.screen = None
                    return None
        elif mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))

    def close(self, **kwargs):
        try:
            super().close(**kwargs)  # Pass kwargs up if superclass can handle them
        except TypeError:
            super().close()  # Fallback if super().close() doesn't take kwargs

        if self.render_initialized and self.screen is not None:
            try:
                if pygame.display.get_init():  # Check if display module is initialized
                    pygame.display.quit()
                if pygame.get_init():  # Check if pygame itself is initialized
                    pygame.quit()
            except Exception as e:
                print(f"Error during pygame quit: {e}")
            self.render_initialized = False
            self.screen = None
            self.clock = None  # Also clear clock
