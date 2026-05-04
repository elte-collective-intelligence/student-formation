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

    def _num_grad_phi(self, p: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        ex = torch.zeros_like(p)
        ex[:, 0] = eps
        ey = torch.zeros_like(p)
        ey[:, 1] = eps
        dx = (self.signed_distance(p + ex) - self.signed_distance(p - ex)) / (2 * eps)
        dy = (self.signed_distance(p + ey) - self.signed_distance(p - ey)) / (2 * eps)
        g = torch.stack([dx, dy], dim=-1)
        return g / torch.norm(g, dim=-1, keepdim=True).clamp(min=1e-8)

    def boundary_frame(self, points: torch.Tensor):
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

    def boundary_frame(self, points: torch.Tensor):
        v = points - self.center
        dist = torch.norm(v, dim=-1, keepdim=True)
        at_center = dist < 1e-6
        n = v / dist.clamp(min=1e-8)
        default_n = torch.tensor(
            [[1.0, 0.0]], device=points.device, dtype=points.dtype
        ).expand_as(n)
        n = torch.where(at_center.expand(-1, 2), default_n, n)
        n = n / torch.norm(n, dim=-1, keepdim=True).clamp(min=1e-8)
        closest = self.center + n * self.radius
        t = torch.stack([-n[:, 1], n[:, 0]], dim=-1)
        return closest, n, t


class Ellipse(Shape):
    def __init__(self, center, semi_axis_x, semi_axis_y, device):
        super().__init__(device)
        self.center = torch.tensor(center, dtype=torch.float32, device=device)
        self.a = float(semi_axis_x)
        self.b = float(semi_axis_y)

    def signed_distance(self, points: torch.Tensor):
        q = points - self.center
        na = q[:, 0] / self.a
        nb = q[:, 1] / self.b
        scale = min(self.a, self.b)
        return (torch.sqrt(na * na + nb * nb + 1e-12) - 1.0) * scale

    def get_target_points(self, num_agents: int) -> torch.Tensor:
        angles = torch.linspace(0, 2 * np.pi, num_agents + 1)[:-1]
        positions = torch.stack(
            [
                self.center[0] + self.a * torch.cos(angles),
                self.center[1] + self.b * torch.sin(angles),
            ],
            dim=1,
        ).to(self.device)
        return positions

    def boundary_frame(self, points: torch.Tensor):
        g = self._num_grad_phi(points)
        s = self.signed_distance(points).unsqueeze(-1)
        closest = points - s * g
        n = self._num_grad_phi(closest)
        t = torch.stack([-n[:, 1], n[:, 0]], dim=-1)
        return closest, n, t


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

    def _closest_on_boundary(self, points: torch.Tensor):
        N = points.shape[0]
        num_v = self.vertices.shape[0]
        best_d2 = torch.full((N,), float("inf"), device=self.device)
        best_p = torch.zeros(N, 2, device=self.device)
        for i in range(num_v):
            v0 = self.vertices[i]
            v1 = self.vertices[(i + 1) % num_v]
            ab = v1 - v0
            ap = points - v0
            t = (ap * ab).sum(dim=-1) / (ab * ab).sum().clamp(min=1e-12)
            t = t.clamp(0.0, 1.0)
            proj = v0 + t.unsqueeze(-1) * ab
            d2 = ((points - proj) ** 2).sum(dim=-1)
            better = d2 < best_d2
            best_d2 = torch.where(better, d2, best_d2)
            best_p = torch.where(better.unsqueeze(-1), proj, best_p)
        return best_p

    def boundary_frame(self, points: torch.Tensor):
        closest = self._closest_on_boundary(points)
        n = self._num_grad_phi(closest)
        t = torch.stack([-n[:, 1], n[:, 0]], dim=-1)
        return closest, n, t

    def get_target_points(self, num_agents: int) -> torch.Tensor:
        """
        Distribute agents along polygon perimeter.
        Works for both convex and non-convex polygons.
        """
        num_vertices = self.vertices.shape[0]

        if num_agents <= num_vertices:
            return self.vertices[:num_agents]

        positions = []

        for i in range(num_agents):
            t = i / num_agents

            edge_idx = t * num_vertices
            v1_idx = int(edge_idx) % num_vertices
            v2_idx = (v1_idx + 1) % num_vertices

            alpha = edge_idx - int(edge_idx)

            v1 = self.vertices[v1_idx]
            v2 = self.vertices[v2_idx]

            pos = (1.0 - alpha) * v1 + alpha * v2
            positions.append(pos)

        result = torch.stack(positions)

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
        self.agent_counts = agent_counts

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        sdfs = []
        for shape in self.shapes:
            sdfs.append(shape.signed_distance(points))

        sdf_stack = torch.stack(sdfs, dim=0)
        return torch.min(sdf_stack, dim=0).values

    def get_target_points(self, num_agents: int) -> torch.Tensor:
        all_targets = []
        for i, shape in enumerate(self.shapes):
            count = self.agent_counts[i]
            pts = shape.get_target_points(count)
            all_targets.append(pts)

        return torch.cat(all_targets, dim=0)

    def boundary_frame(self, points: torch.Tensor):
        sdfs = torch.stack([s.signed_distance(points) for s in self.shapes], dim=0)
        idx = torch.argmin(sdfs, dim=0)
        N = points.shape[0]
        closest = torch.zeros(N, 2, device=self.device, dtype=points.dtype)
        normal = torch.zeros(N, 2, device=self.device, dtype=points.dtype)
        tangent = torch.zeros(N, 2, device=self.device, dtype=points.dtype)
        for k, s in enumerate(self.shapes):
            m = idx == k
            if m.any():
                c, n, t = s.boundary_frame(points[m])
                closest[m] = c
                normal[m] = n
                tangent[m] = t
        return closest, normal, tangent


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
