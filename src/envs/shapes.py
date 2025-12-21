import torch
import math
import numpy as np


class Shape:
    def __init__(self, device):
        self.device = device

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must be implemented in subclasses")

    def get_target_points(self, num_points: int) -> torch.Tensor:
        raise NotImplementedError("Must be implemented in subclasses")


class Circle(Shape):
    def __init__(self, center, radius, device):
        super().__init__(device)
        self.center = torch.tensor(center, dtype=torch.float32, device=device)
        self.radius = radius

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        dist_to_center = torch.norm(points - self.center, dim=-1)
        return dist_to_center - self.radius

    def get_target_points(self, num_agents: int) -> torch.Tensor:
        angles = torch.linspace(0, 2 * np.pi, num_agents + 1)[:-1]
        positions = self.center + self.radius * torch.stack(
            [torch.cos(angles), torch.sin(angles)], dim=1
        ).to(self.device)
        return positions


class Polygon(Shape):
    def __init__(self, vertices, device):
        super().__init__(device)
        self.vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
        self.center = torch.mean(self.vertices, dim=0)

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        num_vertices = self.vertices.shape[0]
        sd_list = []

        for i in range(num_vertices):
            v0 = self.vertices[i]
            v1 = self.vertices[(i + 1) % num_vertices]
            edge = v1 - v0
            edge_normal = torch.tensor([-edge[1], edge[0]], device=self.device)
            edge_normal = edge_normal / torch.norm(edge_normal)

            to_point = points - v0
            proj_length = torch.sum(to_point * edge_normal, dim=-1)
            sd_list.append(proj_length)

        sd_stack = torch.stack(sd_list, dim=-1)
        min_sd = torch.min(sd_stack, dim=-1).values

        # Check if point is inside using ray casting for non-convex support
        def point_in_polygon(p):
            inside = torch.zeros(p.shape[0], dtype=torch.bool, device=self.device)
            for i in range(num_vertices):
                v0 = self.vertices[i]
                v1 = self.vertices[(i + 1) % num_vertices]
                cond = ((v0[1] > p[:, 1]) != (v1[1] > p[:, 1])) & (
                    p[:, 0]
                    < (v1[0] - v0[0]) * (p[:, 1] - v0[1]) / (v1[1] - v0[1]) + v0[0]
                )
                inside ^= cond
            return inside

        is_inside = point_in_polygon(points)
        return torch.where(is_inside, -min_sd, min_sd)

    def get_target_points(self, num_agents: int) -> torch.Tensor:
        """
        Distribute agents along polygon perimeter.
        Works for both convex and non-convex polygons.
        """
        num_vertices = self.vertices.shape[0]

        if num_agents <= num_vertices:
            # If fewer agents than vertices, use first N vertices
            return self.vertices[:num_agents]

        # Otherwise, interpolate along perimeter
        positions = []

        for i in range(num_agents):
            # Normalize position along perimeter [0, 1)
            t = i / num_agents

            # Which edge are we on?
            edge_idx = t * num_vertices
            v1_idx = int(edge_idx) % num_vertices
            v2_idx = (v1_idx + 1) % num_vertices

            # Interpolation factor along edge
            alpha = edge_idx - int(edge_idx)

            v1 = self.vertices[v1_idx]
            v2 = self.vertices[v2_idx]

            # Linear interpolation
            pos = (1.0 - alpha) * v1 + alpha * v2
            positions.append(pos)

        result = torch.stack(positions)

        # Sanity check
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("WARNING: Invalid values in polygon target points.")
            result = self.vertices[
                torch.arange(num_agents, device=self.device) % num_vertices
            ]

        return result


class MultiShape(Shape):
    def __init__(self, shape_list, agent_counts, device):
        super().__init__(device)
        self.shapes = shape_list
        self.agent_counts = agent_counts  # List[int], e.g. [5, 5]

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        # The SDF of a union is the minimum SDF of its parts
        sdfs = []
        for shape in self.shapes:
            sdfs.append(shape.signed_distance(points))

        # Stack [N_shapes, N_agents] and take min
        sdf_stack = torch.stack(sdfs, dim=0)
        return torch.min(sdf_stack, dim=0).values

    def get_target_points(self, num_agents: int) -> torch.Tensor:
        # Concatenate targets from all sub-shapes
        # Note: We ignore 'num_agents' arg here and use the fixed 'agent_counts'
        # defined in the config to ensure stability.
        all_targets = []
        for i, shape in enumerate(self.shapes):
            count = self.agent_counts[i]
            pts = shape.get_target_points(count)
            all_targets.append(pts)

        return torch.cat(all_targets, dim=0)


def make_star_vertices(center, r1, r2, n_points):
    vertices = []
    angle_step = math.pi / n_points
    start_angle = math.pi / 2

    for i in range(2 * n_points):
        r = r2 if i % 2 == 0 else r1
        angle = start_angle + i * angle_step
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        vertices.append([x, y])
    return vertices
