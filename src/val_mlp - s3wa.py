"""Validation with hyperopt"""
import os
import time
from functools import partial
import logging
import torch
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, space_eval
from test_mlp import evaluate_model


def evaluate(config, data_file=None, epochs=1, device=None):
    """Function to be minimized"""
    config = {**config, **config["optimizer_type"]}
    config.pop("optimizer_type", None)
    print(config)
    _, result = evaluate_model(
        csv_file=data_file,
        config=config, 
        epochs=epochs,
        device=device,
    )
    acc = result["mean_gmean"][0]
    return {"loss": -acc, "status": STATUS_OK}


def model_selection(data_file=None, epochs=1, max_evals=1, device=None):
    """Model selection"""
    config = {
        "layers": hp.choice(  # between 1 and 3 layers with random number of hidden nodes
            "layers",
            [
                [hp.randint("l_1", 1, 500)],
                # [hp.randint(f"l_{i}a", 1, 100) for i in range(2)],
                # [hp.randint(f"l_{i}b", 1, 100) for i in range(3)],
            ],
        ),
        "optimizer_type": hp.choice(
            "optimizer_type",
            [
                {
                    "optimizer": "adamw",
                    "beta_1": hp.uniform("adam_beta_1", 0, 1),
                    "beta_2": hp.uniform("adam_beta_2", 0, 1),
                    "lr": hp.uniform("adam_lr", 1e-5, 1),
                    "weight_decay": hp.uniform("adam_weight_decay", 0, 1),
                },
            ],
        ),
        "activation": hp.choice("activation", ["tanh", "logistic", "relu"]),
    }
    func_eval = partial(
        evaluate,
        data_file=data_file,
        epochs=epochs,
        device=device,
    )
    best_config = fmin(
        fn=func_eval, space=config, algo=tpe.suggest, max_evals=max_evals
    )
    # saving
    logging.info("FOUND BEST CONFIG %s", best_config)
    return space_eval(config, best_config)


def main():
    """Main function"""
    logging.basicConfig(
        encoding="utf-8",
        # level=logging.INFO,
        format="[%(levelname)s %(asctime)s] %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S",
    )
    # Reproducibility
    seed = int(time.time())
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Experiment params
    max_evals = 100
    # result analysis setup
    path = os.path.abspath("./")
    config_path = os.path.join(path, "configs/")
    # datasets
    streams = {
        "real": [
            #"sensorClasses29and31",
            #"elec2",
            #"noaa",
            #"powersupplyDayNight",
            #"chess",
            #"luxembourg",
            #"keystroke",
            "rialto",
        ],
        "abrupt": [
            #"Sine",
            #"Agrawal",
            #"SEA",
            #"STAGGER",
        ],
    }
    for data_type, names in streams.items():
        for stream_name in names:
            val_stream = os.path.join(path, f"data/{data_type}/val_{stream_name}.csv")
            print(f"VALIDATING MLP ON {data_type} {stream_name}!!")
            best_config = model_selection(
                data_file=val_stream,
                epochs=1,
                max_evals=max_evals,
                device=device,
            )
            best_config = {
                **best_config,
                **best_config["optimizer_type"],
            }
            best_config.pop("optimizer_type", None)
            np.savez_compressed(
                os.path.join(config_path, f"config_mlp_{stream_name}"),
                best_config=best_config,
                seed=seed,
            )


if __name__ == "__main__":
    main()
