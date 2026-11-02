"""Registers the Unitree R1 walking task with Gymnasium (configs are loaded lazily, after Isaac Sim starts)."""

import gymnasium as gym

gym.register(
    id="R1-Walk-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:R1WalkFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:R1WalkPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="R1-Walk-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:R1WalkFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:R1WalkPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="R1-Walk-Plan-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:R1WalkPlanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:R1WalkPlanPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="R1-Walk-Plan-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:R1WalkPlanEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:R1WalkPlanPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
