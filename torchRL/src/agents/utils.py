def normalize_observations(obs):
    # Example preprocessing step
    return (obs - obs.mean()) / (obs.std() + 1e-8)