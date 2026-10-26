"""PPO settings for R1 walking (same as Isaac Lab's G1 flat-terrain agent)."""

from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class R1WalkPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "r1_walk_flat"
