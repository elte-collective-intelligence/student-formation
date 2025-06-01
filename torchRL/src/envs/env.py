import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.data.tensor_specs import (
    UnboundedContinuousTensorSpec as Unbounded,
    CompositeSpec as Composite,
    BoundedTensorSpec as Bounded,
    DiscreteTensorSpec,
)
import numpy as np
import math
import pygame # For rendering
import pygame.gfxdraw # For anti-aliased shapes

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

        self.num_agents = cfg.env.num_agents
        self.arena_size = cfg.env.arena_size 
        self.max_steps = cfg.env.max_steps
        
        self.agent_size_world_units = cfg.env.get("agent_size", 0.05) # Diameter in world units
        self.agent_accel = cfg.env.get("agent_accel", 3.0) 
        self.agent_max_speed = cfg.env.get("agent_max_speed", 0.3)
        self.dt = 0.1 

        self.agent_positions = torch.zeros(self.num_agents, 2, device=self.device)
        self.agent_velocities = torch.zeros(self.num_agents, 2, device=self.device)
        self.current_step = 0

        # Shape configuration
        self.shape_type = cfg.env.get("shape_type", "circle")
        self.current_target_center = torch.zeros(2, device=self.device) 

        if self.shape_type == "circle":
            self.circle_center = torch.tensor(cfg.env.circle.center, dtype=torch.float32, device=self.device)
            self.circle_radius = cfg.env.circle.radius
            self.current_target_center = self.circle_center
            self.approx_polygon_vertices = None
        elif self.shape_type == "curvilinear_triangle":
            # ... (curvilinear triangle setup - kept for completeness but not focused on now) ...
            ct_cfg = cfg.env.curvilinear_triangle
            self.tri_p1 = torch.tensor(ct_cfg.p1, dtype=torch.float32, device=self.device)
            self.tri_p2 = torch.tensor(ct_cfg.p2, dtype=torch.float32, device=self.device)
            self.tri_p3 = torch.tensor(ct_cfg.p3, dtype=torch.float32, device=self.device)
            self.tri_control1 = torch.tensor(ct_cfg.control1, dtype=torch.float32, device=self.device)
            self.tri_control2 = torch.tensor(ct_cfg.control2, dtype=torch.float32, device=self.device)
            self.num_bezier_points = ct_cfg.get("num_bezier_points", 20)

            self.curve1_points = self._bezier_curve_torch(self.tri_p1, self.tri_control1, self.tri_p2, self.num_bezier_points)
            self.curve2_points = self._bezier_curve_torch(self.tri_p2, self.tri_control2, self.tri_p3, self.num_bezier_points)
            
            self.approx_polygon_vertices = torch.cat([
                self.curve1_points[:-1], 
                self.curve2_points[:-1], 
                self.tri_p3.unsqueeze(0), 
                self.tri_p1.unsqueeze(0)
            ], dim=0)
            self.current_target_center = torch.mean(torch.cat([self.tri_p1.unsqueeze(0), self.tri_p2.unsqueeze(0), self.tri_p3.unsqueeze(0)], dim=0), dim=0)
        else:
            raise ValueError(f"Unsupported shape_type: {self.shape_type}")

        self._make_specs()
        self.actor_obs_keys = cfg.env.obs_keys_for_actor

        # Rendering attributes
        self.screen = None
        self.clock = None
        self.render_scale = 100 # Pixels per world unit (approx)
        self.screen_width = 800
        self.screen_height = 600
        self.render_initialized = False
        self.window_closed_by_user = False


    def _bezier_curve_torch(self, p0_abs, p1_control_abs, p2_abs, num_points=20):
        p0 = p0_abs.unsqueeze(0)
        p1_control = p1_control_abs.unsqueeze(0)
        p2 = p2_abs.unsqueeze(0)
        t = torch.linspace(0, 1, num_points, device=self.device).unsqueeze(1)
        points = ((1 - t)**2 * p0) + (2 * (1 - t) * t * p1_control) + (t**2 * p2)
        return points

    def _make_specs(self) -> None:
        obs_dim_per_agent = 4 
        self.observation_spec = Composite(
            {"observation": Unbounded(shape=(self.num_agents, obs_dim_per_agent), device=self.device)},
            shape=torch.Size([self.num_agents])
        )
        self.action_spec = Bounded(
            low=-1.0, high=1.0, shape=(self.num_agents, 2),
            device=self.device, dtype=torch.float32
        )
        self.reward_spec_unbatched = Unbounded(shape=(1,), device=self.device)
        self.done_spec_unbatched = DiscreteTensorSpec(
            n=2, shape=torch.Size([1]), dtype=torch.bool, device=self.device
        )

    def _get_observations(self) -> torch.Tensor:
        observations = torch.zeros(self.num_agents, 4, device=self.device)
        vec_to_center = self.current_target_center.unsqueeze(0) - self.agent_positions
        dist_matrix = torch.cdist(self.agent_positions, self.agent_positions)
        dist_matrix.fill_diagonal_(float('inf'))
        if self.num_agents > 1:
            _, closest_indices = torch.min(dist_matrix, dim=1)
            closest_agent_positions = self.agent_positions[closest_indices]
            vec_to_closest = closest_agent_positions - self.agent_positions
        else:
            vec_to_closest = torch.zeros_like(self.agent_positions)
        observations[:, :2] = vec_to_center
        observations[:, 2:] = vec_to_closest
        return observations

    def _reset(self, tensordict: TensorDictBase = None) -> TensorDictBase:
        self.current_step = 0
        min_val, max_val = -self.arena_size / 2, self.arena_size / 2
        self.agent_positions = (torch.rand(self.num_agents, 2, device=self.device) * (max_val - min_val)) + min_val
        self.agent_velocities = torch.zeros(self.num_agents, 2, device=self.device)
        current_observations = self._get_observations()
        done_val = torch.zeros((self.num_agents, 1), dtype=torch.bool, device=self.device)
        return TensorDict({
                "observation": current_observations,
                "done": done_val,
            }, batch_size=torch.Size([self.num_agents]), device=self.device)

    def _point_in_polygon_torch_batched(self, points: torch.Tensor, polygon_vertices: torch.Tensor) -> torch.Tensor:
        if polygon_vertices is None or polygon_vertices.shape[0] < 3:
             print_warning_once(f"Polygon for {self.shape_type} is not defined. Defaulting to False for point_in_polygon.")
             return torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        inside = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        for i in range(points.shape[0]):
            p_x, p_y = points[i, 0], points[i, 1]
            crossings = 0
            for j in range(polygon_vertices.shape[0]):
                v1_x, v1_y = polygon_vertices[j, 0], polygon_vertices[j, 1]
                v2_x, v2_y = polygon_vertices[(j + 1) % polygon_vertices.shape[0], 0], polygon_vertices[(j + 1) % polygon_vertices.shape[0], 1]
                if (((v1_y <= p_y) and (v2_y > p_y)) or ((v1_y > p_y) and (v2_y <= p_y))):
                    if p_x < (v2_x - v1_x) * (p_y - v1_y) / (v2_y - v1_y + 1e-9) + v1_x :
                        crossings += 1
            if crossings % 2 == 1:
                inside[i] = True
        return inside

    def _point_in_shape_batch(self, points_to_check: torch.Tensor) -> torch.Tensor:
        if self.shape_type == "circle":
            # For circle, "in_shape" can mean being close to the circumference
            # or simply inside. Let's use "inside" for now for boundary penalty.
            dist_to_center = torch.norm(points_to_check - self.circle_center.unsqueeze(0), dim=1)
            return dist_to_center <= self.circle_radius # True if inside or on the circle
        elif self.shape_type == "curvilinear_triangle":
            return self._point_in_polygon_torch_batched(points_to_check, self.approx_polygon_vertices)
        return torch.zeros(points_to_check.shape[0], dtype=torch.bool, device=self.device)

    def _calc_rewards(self) -> torch.Tensor:
        rewards = torch.zeros(self.num_agents, 1, device=self.device)

        # Reward for being at the correct distance from the circle center
        if self.shape_type == "circle":
            dist_to_center = torch.norm(self.agent_positions - self.circle_center.unsqueeze(0), dim=1, keepdim=True)
            # Gaussian-like reward for being at self.circle_radius distance
            # Higher reward when dist_to_center is close to self.circle_radius
            target_dist_error = torch.abs(dist_to_center - self.circle_radius)
            # Use a negative exponential: reward = exp(-k * error^2)
            # Smaller error -> reward closer to 1. Larger error -> reward closer to 0.
            # k controls sensitivity. Let's try k=5 for now.
            formation_accuracy_reward = torch.exp(-5.0 * target_dist_error**2)
            rewards += formation_accuracy_reward

            # Optional: Spacing reward (can be complex)
            # For now, let's add a small penalty for being too close to other agents
            if self.num_agents > 1:
                dist_matrix = torch.cdist(self.agent_positions, self.agent_positions)
                dist_matrix.fill_diagonal_(float('inf')) # Ignore self-distance
                closest_dists, _ = torch.min(dist_matrix, dim=1, keepdim=True)
                # Penalize if too close, e.g., closer than 2 * agent_size
                min_spacing = 2.0 * self.agent_size_world_units
                spacing_penalty = torch.clamp(min_spacing - closest_dists, min=0.0) * 0.5 # Scale penalty
                rewards -= spacing_penalty


        elif self.shape_type == "curvilinear_triangle":
            # Original reward logic from your MPE environment for being in shape
            in_shape = self._point_in_shape_batch(self.agent_positions)
            if self.num_agents > 1:
                dist_matrix = torch.cdist(self.agent_positions, self.agent_positions)
                dist_matrix.fill_diagonal_(float('inf'))
                closest_dists_to_others, _ = torch.min(dist_matrix, dim=1)
                # Only apply this part of reward if in_shape is True
                rewards[in_shape, 0] += closest_dists_to_others[in_shape] 
            elif self.num_agents == 1: # Single agent
                 rewards[in_shape, 0] += 0.1 # Small positive reward for being in shape alone

        # Boundary penalty (from your MPE code, applied universally)
        # This assumes arena is roughly [-1, 1] if boundary is 0.9
        # We should use self.arena_size for boundary checks.
        normalized_pos = self.agent_positions / (self.arena_size / 2.0) # Normalize to roughly [-1, 1]
        abs_norm_pos = torch.abs(normalized_pos)
        
        penalty = torch.zeros_like(abs_norm_pos) # [N, 2]
        # Thresholds for penalty based on normalized position
        bound_thresh_soft = 0.95 # Start penalty slightly inside the arena edge
        bound_thresh_hard = 1.0  # Max penalty at/beyond arena edge

        cond2 = (abs_norm_pos >= bound_thresh_soft) & (abs_norm_pos < bound_thresh_hard)
        penalty[cond2] = (abs_norm_pos[cond2] - bound_thresh_soft) * 20 # Scaled penalty
        
        cond3 = abs_norm_pos >= bound_thresh_hard
        # Stronger penalty if outside the hard boundary
        penalty[cond3] = torch.min(torch.exp(5 * (abs_norm_pos[cond3] - bound_thresh_hard)), torch.tensor(10.0, device=self.device)) + \
                         (abs_norm_pos[cond3] - bound_thresh_hard) * 20 # Ensure it grows

        total_penalty_per_agent = torch.sum(penalty, dim=1, keepdim=True) # [N, 1]
        rewards -= total_penalty_per_agent
        
        return rewards

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.current_step += 1
        actions = tensordict["action"] 
        force = actions * self.agent_accel
        self.agent_velocities += force * self.dt
        speed = torch.norm(self.agent_velocities, dim=1, keepdim=True)
        too_fast_mask = speed > self.agent_max_speed
        safe_speed = speed.clone()
        safe_speed[speed == 0] = 1e-6
        mask_for_vel_update = too_fast_mask.squeeze(-1)
        if mask_for_vel_update.any():
             self.agent_velocities[mask_for_vel_update] = \
                (self.agent_velocities[mask_for_vel_update] / safe_speed[mask_for_vel_update]) * self.agent_max_speed
        self.agent_positions += self.agent_velocities * self.dt
        
        # Keep agents within defined arena_size (hard clamp)
        self.agent_positions = torch.clamp(self.agent_positions, -self.arena_size/2, self.arena_size/2)

        current_rewards = self._calc_rewards()
        is_episode_done = self.current_step >= self.max_steps
        done_val = torch.full((self.num_agents, 1), is_episode_done, dtype=torch.bool, device=self.device)
        next_observations = self._get_observations()
        return TensorDict({
                "observation": next_observations, "reward": current_rewards, "done": done_val,
            }, batch_size=torch.Size([self.num_agents]), device=self.device)

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
        screen_span_w = self.screen_width * 0.9 # Use 90% of screen for margin
        screen_span_h = self.screen_height * 0.9
        self.render_scale = min(screen_span_w / world_span, screen_span_h / world_span)
        self.render_offset_x = self.screen_width / 2
        self.render_offset_y = self.screen_height / 2
        self.render_initialized = True
        self.window_closed_by_user = False


    def _to_screen_coords(self, world_pos_tensor): # world_pos_tensor is [N, 2] or [2]
        # World origin (0,0) maps to screen center.
        # Pygame y is inverted.
        screen_pos = world_pos_tensor.clone()
        screen_pos[:, 1] *= -1 # Invert y-axis for Pygame
        screen_pos *= self.render_scale
        screen_pos += torch.tensor([self.render_offset_x, self.render_offset_y], device=self.device)
        return screen_pos.cpu().numpy().astype(int) # Return as int numpy array

    def render(self, mode="human"):
        if self.window_closed_by_user: # If user closed window, don't try to render
             if mode == "rgb_array": # Still need to return an array for GIF
                 return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
             return None


        if not self.render_initialized and mode == "human":
            self._init_render()
        elif mode == "rgb_array" and self.screen is None:
            self._init_render()
            # pygame.display.iconify() # Optional: hide window if only for rgb_array

        if self.screen is None and mode == "human": # Should have been initialized
             self._init_render()
        elif self.screen is None and mode == "rgb_array":
             self._init_render() # Make sure screen is available for surfarray

        self.screen.fill((255, 255, 255))

        # --- Draw Target Shape ---
        if self.shape_type == "circle":
            s_center = self._to_screen_coords(self.circle_center.unsqueeze(0))[0]
            s_radius = int(self.circle_radius * self.render_scale)
            pygame.gfxdraw.aacircle(self.screen, s_center[0], s_center[1], s_radius, (200, 200, 200))
            pygame.gfxdraw.filled_circle(self.screen, s_center[0], s_center[1], s_radius, (220, 220, 220, 100)) # Slightly transparent fill
        elif self.shape_type == "curvilinear_triangle":
            if self.curve1_points is not None and self.curve2_points is not None:
                curve1_sc = self._to_screen_coords(self.curve1_points)
                curve2_sc = self._to_screen_coords(self.curve2_points)
                p1_sc = self._to_screen_coords(self.tri_p1.unsqueeze(0))[0]
                p3_sc = self._to_screen_coords(self.tri_p3.unsqueeze(0))[0]
                
                if len(curve1_sc) > 1: pygame.draw.aalines(self.screen, (180, 180, 255), False, curve1_sc.tolist())
                if len(curve2_sc) > 1: pygame.draw.aalines(self.screen, (180, 180, 255), False, curve2_sc.tolist())
                pygame.draw.aaline(self.screen, (180, 180, 255), tuple(p1_sc), tuple(p3_sc))

        # --- Draw Agents ---
        agent_screen_pos = self._to_screen_coords(self.agent_positions)
        s_agent_radius = int(self.agent_size_world_units * self.render_scale / 2) 
        s_agent_radius = max(s_agent_radius, 2) 

        for i in range(self.num_agents):
            color = (70, 180, 70) 
            pos = tuple(agent_screen_pos[i])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], s_agent_radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], s_agent_radius, color)
            # pygame.draw.circle(self.screen, (0,0,0), pos, s_agent_radius, 1) # Border

        if mode == "human":
            pygame.display.flip()
            if self.clock: self.clock.tick(self.metadata["render_fps"])
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
            super().close(**kwargs) # Pass kwargs up if superclass can handle them
        except TypeError:
            super().close() # Fallback if super().close() doesn't take kwargs

        if self.render_initialized and self.screen is not None:
            try:
                if pygame.display.get_init(): # Check if display module is initialized
                    pygame.display.quit()
                if pygame.get_init(): # Check if pygame itself is initialized
                    pygame.quit() 
            except Exception as e:
                print(f"Error during pygame quit: {e}")
            self.render_initialized = False
            self.screen = None
            self.clock = None # Also clear clock