"""MLG Water Bucket Gym"""

from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.mc import MS_PER_STEP, STEPS_PER_MS
from minerl.herobraine.hero.handler import Handler
from typing import List

from minerl.herobraine.hero.mc import ALL_ITEMS

import minerl.herobraine.hero.handlers as handlers

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

MLGWB_DOC = """
In MLG Water Bucket, an agent must learn to perform an "MLG Water Bucket"
"""

MLGWB_LENGTH = 8000

class MLGWB(SimpleEmbodimentEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MLGWB-v0'

        super().__init__(*args,
                        max_episode_steps=MLGWB_LENGTH, reward_threshold=100.0,
                        **kwargs)

    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.FlatWorldGenerator(generatorString="1;7,2x3,2;1"),
            handlers.DrawingDecorator("""
                <DrawCuboid x1="5" y1="5" z1="2" x2="5" y2="5" z2="2" type="gold_block"/>
                <DrawCuboid x1="-2" y1="88" z1="-2" x2="2" y2="88" z2="2" type="obsidian"/>
            """)
        ]

    def create_agent_start(self) -> List[Handler]:
        return [
            handlers.SimpleInventoryAgentStart([
                dict(type="water_bucket", quantity=1)
            ]),
            handlers.AgentStartPlacement(0, 90, 0, 0, 0)
        ]

    def create_rewardables(self) -> List[Handler]:
        return [
            handlers.RewardForTouchingBlockType([
                {'type':'gold_block', 'behaviour':'onceOnly', 'reward':'100'},
            ])
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return [
            handlers.AgentQuitFromTouchingBlockType([
                "gold_block"
            ])
        ]
    
    
    def create_actionables(self) -> List[Handler]:
        return super().create_actionables() + [
            # allow agent to place water
            handlers.PlaceBlock(['none', 'water_bucket'],
                                _other='none', _default='none')

        ]

    def create_observables(self) -> List[Handler]:
        return super().create_observables() + [
            # A compass observation which returns angle and distance information
            handlers.CompassObservation(True, True),
        ]
    
    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            # Sets time to morning and stops passing of time
            handlers.TimeInitialCondition(False, 23000)
        ]

    def create_server_quit_producers(self):
        return []
    
    def create_server_decorators(self) -> List[Handler]:
        return []

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'mlgwb'

    def get_docstring(self):
        return MLGWB_DOC
