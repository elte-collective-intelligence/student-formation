# Formation Control Report

This project extends the baseline TorchRL formation task into a reproducible study on multi-agent formation control. The environment supports multiple target shapes, multishape scenes, dynamic reconfiguration, Hungarian and greedy assignment, SDF-based observations, evaluation metrics, tests, charts, and GIF visualization.

The main task uses 10 agents in a multishape scene. Agents first form a circle and polygon setup, then the target changes during the episode to a polygon and circle. Training uses PPO for 1,000,000 frames per run on CPU. Each ablation was run with seeds 0, 1, and 2.

For the assignment-method ablation, Hungarian assignment performed better overall than greedy assignment. It reduced mean boundary error from 3.651 to 3.189 and reduced collision rate from 22.21% to 2.24%. Greedy had slightly higher training reward, but its collision rate was much worse, so Hungarian is the stronger choice for this task.

For the observation ablation, using SDF geometry features improved mean boundary error from 3.570 to 3.189. The no-SDF setting had zero collisions and higher reward in this run, so the result shows a clear trade-off: SDF observations help agents fit the target boundary better, while the simpler observation setting is more stable for reward and collisions.

Overall, the project delivers a solid reproducible MARL formation-control study with the requested extensions. The results support Hungarian assignment as the best assignment strategy in this setup and show that SDF observations improve boundary accuracy. The controller is not a fully solved formation policy, since boundary error remains visible, but the implementation, ablations, and qualitative rollout give a defensible and coherent final submission.

The final artefacts are:

- `runs_assignment.csv`
- `runs_geometry_obs.csv`
- `chart_assignment.png`
- `chart_geometry_obs.png`
- `formation.gif`

Verification: the test suite passed with `11 passed, 1 warning`. The warning is only a dependency deprecation warning from `pygame`, not a project failure.
