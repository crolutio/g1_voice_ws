"""PPO settings for R1 walking (same as Isaac Lab's G1 flat-terrain agent)."""

from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class R1WalkPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "r1_walk_flat"
        # Less exploration noise than the G1 default (0.008): with the many gait rules the action noise kept
        # growing (std 0.6 -> 1.8) until training became unstable.
        self.algorithm.entropy_coef = 0.002


@configclass
class R1WalkPlanPPORunnerCfg(R1WalkPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "r1_walk_plan"
        # fine-tuning a policy that already walks: small, careful updates so it keeps what it knows
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.desired_kl = 0.005
