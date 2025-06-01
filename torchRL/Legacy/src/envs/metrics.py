def compute_spacing_variance(positions):
    dists = (positions.unsqueeze(1) - positions.unsqueeze(0)).norm(dim=2)
    return torch.var(dists)

def compute_obstacle_penalty(observations):
    return torch.tensor(0.0)

def compute_formation_completion_time(step_count, threshold=0.01):
    return step_count if threshold < 0.01 else 0
