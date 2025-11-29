import torch
import math


class Shape:
    def __init__(self, device):
        self.device = device

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Must be implemented in subclasses")


class Circle(Shape):
    def __init__(self, center, radius, device):
        super().__init__(device)
        self.center = torch.tensor(center, dtype=torch.float32, device=device)
        self.radius = radius

    def signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        dist_to_center = torch.norm(points - self.center, dim=-1)
        return dist_to_center - self.radius


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
        return torch.min(sd_stack, dim=-1).values


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
