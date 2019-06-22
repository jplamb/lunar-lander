import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print
import numpy as np

ALGOS = {
    'ppo': ppo.PPOTrainer,
    'dqn': dqn.DQNTrainer
}

CONFIGS = {
    'ppo': ppo.DEFAULT_CONFIG.copy(),
    'dqn': dqn.DEFAULT_CONFIG.copy()
}


class Trainer:
    def __init__(self,
                 num_iterations,
                 exp_name='',
                 env_name='LunarLanderContinuous-v2',
                 algo='PPO',
                 checkpoint_freq=100,
                 evaluation_interval=100,
                 evaluation_num_episodes=10,
                 num_gpus=0,
                 num_workers=1):
        self.env_name = env_name
        self.exp_name = exp_name
        self.algo = algo.lower()
        self.num_gpus = num_gpus
        self.num_workers = num_workers
        self.checkpoint_freq = checkpoint_freq
        self.num_iterations = num_iterations
        self.evaluation_interval = evaluation_interval
        self.evaluation_num_episodes = evaluation_num_episodes

        ray.init(temp_dir=f'./results/{self.exp_name}')

    def train(self):
        config = CONFIGS[self.algo]
        config['num_gpus'] = self.num_gpus
        config['num_workers'] = self.num_workers
        config['monitor'] = True
        config['evaluation_interval'] = self.evaluation_interval
        config['evaluation_num_episodes'] = self.evaluation_num_episodes

        if self.algo not in ALGOS:
            raise NotImplementedError(f'Unrecognized algorithm {self.algo}')

        agent = ALGOS[self.algo](config=config, env=self.env_name)

        for i in range(self.num_iterations):
            result = agent.train()
            print(pretty_print(result))

            if i % self.checkpoint_freq == 0:
                checkpoint = agent.save()
                print(f'Saving checkpoint at {checkpoint} loops')

    def tune(self, stop_reward_mean=150):
        if self.algo != 'ppo':
            raise NotImplementedError(f'{self.algo} not implemented')
        tune.run(self.algo.upper(),
                 stop={'episode_reward_mean': stop_reward_mean, 'training_iteration': 100},
                 verbose=1,
                 num_samples=10,
                 name=self.exp_name,
                 config={
                     'env': self.env_name,
                     'num_gpus': self.num_gpus,
                     'num_workers': self.num_workers,
                     'lr': tune.sample_from(lambda spec: np.random.choice([0.01, 0.001, 0.0001])),
                     'sample_batch_size': tune.sample_from(lambda spec: np.random.choice([64, 128, 256, 512])),
                     'lambda': tune.sample_from(lambda spec: np.random.choice([.5, .6, .7, .8, .9, 1.])),
                     'clip_param': tune.sample_from(lambda spec: np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5]))
                 })
