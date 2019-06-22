from trainer import Trainer
from resources.human_id import generate_human_readable_id

TUNE_PRIOR = False


def run(tune_prior):
    exp_name = generate_human_readable_id()
    trainer = Trainer(
        exp_name=exp_name,
        algo='dqn',
        num_iterations=1000,
        num_workers=7,
        env_name='LunarLander-v2'
    )

    if tune_prior:
        trainer.tune()
    else:
        trainer.train()


if __name__ == "__main__":
    run(TUNE_PRIOR)
