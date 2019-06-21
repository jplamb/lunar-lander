from trainer import Trainer

TUNE_PRIOR = True


def run(tune_prior):
    trainer = Trainer(num_iterations=1000)

    if tune_prior:
        trainer.tune()
    trainer.train()


if __name__ == "__main__":
    run(TUNE_PRIOR)
