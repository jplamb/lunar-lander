import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print


class Trainer:
    def __init__(self,
                 num_iterations,
                 env_name='LunarLanderContinuous-v2',
                 algo='PPO',
                 checkpoint_freq=100,
                 evaluation_interval=100,
                 evaluation_num_episodes=10,
                 num_gpus=0,
                 num_workers=1):
        self.env_name = env_name
        self.algo = algo
        self.num_gpus = num_gpus
        self.num_workers = num_workers
        self.checkpoint_freq = checkpoint_freq
        self.num_iterations = num_iterations
        self.evaluation_interval = evaluation_interval
        self.evaluation_num_episodes = evaluation_num_episodes

        ray.init(temp_dir='./results/')

    def train(self):
        if self.algo != 'ppo':
            raise NotImplementedError(f'{self.algo} not implemented')
        config = ppo.DEFAULT_CONFIG.copy()
        config['num_gpus'] = self.num_gpus
        config['num_workers'] = self.num_workers
        config['monitor'] = True
        config['evaluation_interval'] = self.evaluation_interval
        config['evaluation_num_episodes'] = self.evaluation_num_episodes

        agent = ppo.PPOTrainer(config=config, env=self.env_name)

        for i in range(self.num_iterations):
            result = agent.train()
            print(pretty_print(result))

            if i % self.checkpoint_freq == 0:
                checkpoint = agent.save()
                print(f'Saving checkpoint at {checkpoint} loops')

    def tune(self, stop_reward_mean=150):
        tune.run(self.algo.upper(),
                 stop={'episode_reward_mean': stop_reward_mean},
                 config={
                     'env': self.env_name,
                     'num_gpus': self.num_gpus,
                     'num_workers': self.num_workers,
                     'lr': tune.grid_search([0.01, 0.001, 0.0001])
                 })
