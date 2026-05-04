import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.data.tensor_specs import (
    BoundedContinuous,
    Composite,
    UnboundedContinuous,
)

try:
    from torchrl.data.tensor_specs import DiscreteTensorSpec
except ImportError:
    DiscreteTensorSpec = None


def _make_done_spec(device):
    if DiscreteTensorSpec is not None:
        return DiscreteTensorSpec(
            n=2, shape=torch.Size([1]), dtype=torch.bool, device=device
        )
    from torchrl.data.tensor_specs import Binary

    return Binary(shape=torch.Size([1]), dtype=torch.bool, device=device)


from scipy.optimize import linear_sum_assignment
import numpy as np
import pygame
import pygame.gfxdraw

from src.envs.shapes import (
    Circle,
    Ellipse,
    MultiShape,
    Polygon,
    make_star_vertices,
)

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
        )
        self.agent_size = self.agent_size_world_units
        self.agent_accel = cfg.env.get("agent_accel", 3.0)
        self.agent_max_speed = cfg.env.get("agent_max_speed", 0.3)
        self.dt = 0.1

        self.agent_positions = torch.zeros(self.num_agents, 2, device=self.device)
        self.agent_velocities = torch.zeros(self.num_agents, 2, device=self.device)
        self.current_step = 0

        self.reconfig_step = cfg.env.get("reconfig_step", None)
        self.has_reconfigured = False

        self.shape_type = cfg.env.get("shape_type", "circle")
        self.target_shape = self.__create_shape(self.shape_type, cfg.env)

        self.assignment_method = cfg.env.get("assignment_method", "greedy")
        self.use_sdf_obs = cfg.env.get("use_sdf_obs", True)
        self.knn_k = int(cfg.env.get("knn_k", 3))
        self.radial_velocity_penalty_weight = float(
            cfg.env.get("radial_velocity_penalty_weight", 0.12)
        )
        self.shape_boundary_points = self.target_shape.get_target_points(
            self.num_agents
        )
        self.assigned_target_positions = torch.zeros(
            self.num_agents, 2, device=self.device
        )

        self._make_specs()
        self.actor_obs_keys = cfg.env.obs_keys_for_actor

        self.screen = None
        self.clock = None
        self.render_scale = 100
        self.screen_width = 800
        self.screen_height = 600
        self.render_initialized = False
        self.window_closed_by_user = False

    def __create_shape(self, shape_type, cfg):
        if shape_type == "circle":
            c_cfg = cfg.circle
            return Circle(c_cfg.center, c_cfg.radius, self.device)

        elif shape_type == "ellipse":
            e_cfg = cfg.ellipse
            return Ellipse(
                e_cfg.center,
                float(e_cfg.semi_axis_x),
                float(e_cfg.semi_axis_y),
                self.device,
            )

        elif shape_type == "polygon":
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
            for s_cfg in cfg.multishape:
                t = s_cfg.type
                counts.append(s_cfg.agent_count)

                if t == "circle":
                    s = Circle(s_cfg.center, s_cfg.radius, self.device)
                elif t == "ellipse":
                    s = Ellipse(
                        s_cfg.center,
                        float(s_cfg.semi_axis_x),
                        float(s_cfg.semi_axis_y),
                        self.device,
                    )
                elif t == "polygon":
                    v = torch.tensor(
                        s_cfg.vertices, device=self.device, dtype=torch.float32
                    )
                    s = Polygon(v, device=self.device)
                elif t == "star":
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
            new_cfg = self.cfg.env.reconfig_shape
            new_type = new_cfg.shape_type

            self.shape_type = new_type
            self.target_shape = self.__create_shape(new_type, new_cfg)

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

    def _obs_feature_dim(self) -> int:
        knn_dim = max(0, min(self.knn_k, max(0, self.num_agents - 1)))
        if self.use_sdf_obs:
            return 1 + 2 + 2 + 2 + 2 + knn_dim
        return 2 + 2 + knn_dim

    def _make_specs(self) -> None:
        obs_dim_per_agent = self._obs_feature_dim()
        self.observation_spec = Composite(
            {
                "observation": UnboundedContinuous(
                    shape=(self.num_agents, obs_dim_per_agent), device=self.device
                )
            },
            shape=torch.Size([self.num_agents]),
        )
        self.action_spec = BoundedContinuous(
            low=-1.0,
            high=1.0,
            shape=(self.num_agents, 2),
            device=self.device,
            dtype=torch.float32,
        )
        self.reward_spec_unbatched = UnboundedContinuous(shape=(1,), device=self.device)
        self.done_spec_unbatched = _make_done_spec(self.device)

    def _knn_distances(self) -> torch.Tensor:
        knn_dim = max(0, min(self.knn_k, max(0, self.num_agents - 1)))
        if knn_dim == 0:
            return torch.zeros(self.num_agents, 0, device=self.device)
        dist_matrix = torch.cdist(self.agent_positions, self.agent_positions)
        dist_matrix.fill_diagonal_(float("inf"))
        knearest, _ = torch.topk(
            dist_matrix, knn_dim, dim=1, largest=False, sorted=True
        )
        return knearest

    def _get_observations(self) -> torch.Tensor:
        target_vec = self.assigned_target_positions - self.agent_positions
        knn_d = self._knn_distances()

        if self.use_sdf_obs:
            sdf = self.target_shape.signed_distance(self.agent_positions).unsqueeze(1)
            _, normal, tangent = self.target_shape.boundary_frame(self.agent_positions)
            vel = self.agent_velocities
            observations = torch.cat(
                [sdf, normal, tangent, vel, target_vec, knn_d], dim=1
            )
        else:
            vel = self.agent_velocities
            observations = torch.cat([vel, target_vec, knn_d], dim=1)

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

        _, normal, _ = self.target_shape.boundary_frame(self.agent_positions)
        v_rad = (self.agent_velocities * normal).sum(dim=-1, keepdim=True)
        rewards -= self.radial_velocity_penalty_weight * (v_rad**2)

        dist_to_assigned = torch.norm(
            self.assigned_target_positions - self.agent_positions, dim=1, keepdim=True
        )
        assignment_accuracy_reward = torch.exp(-2.0 * dist_to_assigned**2)
        rewards += assignment_accuracy_reward

        normalized_pos = self.agent_positions / (
            self.arena_size / 2.0
        )
        abs_norm_pos = torch.abs(normalized_pos)

        penalty = torch.zeros_like(abs_norm_pos)
        bound_thresh_soft = 0.95
        bound_thresh_hard = 1.0

        cond2 = (abs_norm_pos >= bound_thresh_soft) & (abs_norm_pos < bound_thresh_hard)
        penalty[cond2] = (
            abs_norm_pos[cond2] - bound_thresh_soft
        ) * 20

        cond3 = abs_norm_pos >= bound_thresh_hard
        penalty[cond3] = (
            torch.min(
                torch.exp(5 * (abs_norm_pos[cond3] - bound_thresh_hard)),
                torch.tensor(10.0, device=self.device),
            )
            + (abs_norm_pos[cond3] - bound_thresh_hard) * 20
        )

        total_penalty_per_agent = torch.sum(penalty, dim=1, keepdim=True)
        rewards -= total_penalty_per_agent

        return rewards

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.current_step += 1

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
        world_span = self.arena_size
        screen_span_w = self.screen_width * 0.9
        screen_span_h = self.screen_height * 0.9
        self.render_scale = min(screen_span_w / world_span, screen_span_h / world_span)
        self.render_offset_x = self.screen_width / 2
        self.render_offset_y = self.screen_height / 2
        self.render_initialized = True
        self.window_closed_by_user = False

    def _to_screen_coords(self, world_pos_tensor):
        screen_pos = world_pos_tensor.clone()
        screen_pos[:, 1] *= -1
        screen_pos *= self.render_scale
        screen_pos += torch.tensor(
            [self.render_offset_x, self.render_offset_y], device=self.device
        )
        return screen_pos.cpu().numpy().astype(int)

    def render(self, mode="human"):
        if self.window_closed_by_user:
            if mode == "rgb_array":
                return np.zeros(
                    (self.screen_height, self.screen_width, 3), dtype=np.uint8
                )
            return None

        if not self.render_initialized and mode == "human":
            self._init_render()
        elif mode == "rgb_array" and self.screen is None:
            self._init_render()

        if self.screen is None and mode == "human":
            self._init_render()
        elif self.screen is None and mode == "rgb_array":
            self._init_render()

        self.screen.fill((255, 255, 255))

        def draw_one(s):
            if isinstance(s, Circle):
                c = self._to_screen_coords(s.center.unsqueeze(0))[0]
                r = int(s.radius * self.render_scale)
                pygame.gfxdraw.aacircle(self.screen, c[0], c[1], r, (200, 200, 200))
            elif isinstance(s, Ellipse):
                c = self._to_screen_coords(s.center.unsqueeze(0))[0]
                rx = max(int(s.a * self.render_scale), 1)
                ry = max(int(s.b * self.render_scale), 1)
                pygame.draw.ellipse(
                    self.screen,
                    (200, 200, 200),
                    (c[0] - rx, c[1] - ry, 2 * rx, 2 * ry),
                    width=1,
                )
            elif isinstance(s, Polygon):
                v = self._to_screen_coords(s.vertices)
                pygame.draw.aalines(self.screen, (200, 200, 200), True, v.tolist())

        if isinstance(self.target_shape, MultiShape):
            for sub_s in self.target_shape.shapes:
                draw_one(sub_s)
        else:
            draw_one(self.target_shape)

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

        if mode == "human":
            pygame.display.flip()
            if self.clock:
                self.clock.tick(self.metadata["render_fps"])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.window_closed_by_user = True
                    return None
        elif mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))

    def close(self, **kwargs):
        try:
            super().close(**kwargs)
        except TypeError:
            super().close()

        if self.render_initialized and self.screen is not None:
            try:
                if pygame.display.get_init():
                    pygame.display.quit()
                if pygame.get_init():
                    pygame.quit()
            except Exception as e:
                print(f"Error during pygame quit: {e}")
            self.render_initialized = False
            self.screen = None
            self.clock = None
