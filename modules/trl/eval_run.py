import fire
from perl.eval import evaluate, EvalConfig


def main(**kwargs):
    config = EvalConfig(**kwargs)
    evaluate(config)


if __name__ == "__main__":
    fire.Fire(main)
