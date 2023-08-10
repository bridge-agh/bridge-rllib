from gymnasium import spaces
import tensorflow as tf

import ray

from ray import tune
from ray.tune.registry import register_env

from ray.air.integrations.wandb import WandbLoggerCallback

from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork

from bridge_rllib.envs.bridge_v0 import BridgeEnv


class ActionMaskModel(TFModelV2):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, spaces.Dict)
            and "action_mask" in orig_space.spaces
            and "observation" in orig_space.spaces
        )

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observation"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observation"]})
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


def run():
    ray.init()

    register_env('Bridge-v0', lambda env_config: BridgeEnv(**env_config))

    n_steps = 32
    n_workers = 23
    n_envs_per_worker = 64

    batch_size = n_steps * n_envs_per_worker * n_workers

    config = (
        APPOConfig()
        .training(
            train_batch_size=batch_size,
            replay_proportion=1,
            model={
                "custom_model": ActionMaskModel,
            },
        )
        .framework('tf')
        .environment(
            env='Bridge-v0',
            env_config={
                'reward_mode': 'sparse',
                'render_mode': None,
            },
        )
        .rollouts(
            rollout_fragment_length=n_steps,
            num_rollout_workers=n_workers,
            num_envs_per_worker=n_envs_per_worker,
        )
        .resources(
            num_gpus=0.5,
            num_gpus_per_worker=0.49 / n_workers,
        )
    )

    tune.run(
        'APPO',
        config=config.to_dict(),
        callbacks=[WandbLoggerCallback(project='bridge')],
        local_dir='./ray_results',
        stop={
            'time_total_s': 60 * 60 * 4,
        },
    )

    ray.shutdown()


if __name__ == '__main__':
    run()
