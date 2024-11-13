from gymnasium.envs.registration import register


register(
    id='cable_bridge_di-v0',
    entry_point='bridgezoo.cablebridge_di.envs.cablebridge_base:CableBridgeBase',
    max_episode_steps=200,
)
