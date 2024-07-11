"""Validation with hyperopt"""
import os
import time
from functools import partial
import logging
import torch
import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, space_eval
from test_s3wa import evaluate_model


def evaluate(config, data_file=None, base_config=None, epochs=1, device=None):
    """Function to be minimized"""
    config = {**config, **base_config}
    print(config)
    mask_path = os.path.join(f"C:\\Users\\dougl\\OneDrive\\Documentos\\S3WA\\data\\real\\lab_val_rialto_uni_l10.csv")
    _, result = evaluate_model(
        csv_file=data_file,
        config=config,
        mask_file=mask_path,
        epochs=epochs,
        device=device,
    )
    acc = result["mean_gmean"][0]
    return {"loss": -acc, "status": STATUS_OK}


def model_selection(
    data_file=None, base_config=None, epochs=1, max_evals=1, device=None
):
    """Model selection"""
    n_instances = np.loadtxt(data_file, delimiter=",").shape[0]
    config = {
        "start": hp.randint("start", 2, int(0.5 * n_instances)),
        "alpha_zero": hp.uniform("eta", 1e-5, 1 - 1e-5),
    }
    func_eval = partial(
        evaluate,
        data_file=data_file,
        base_config=base_config,
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
    torch.manual_seed(seed)
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
            print(f"VALIDATING CWA ON {data_type} {stream_name}!!")
            base_config = np.load(
                os.path.join(config_path, f"config_mlp_{stream_name}.npz"),
                allow_pickle=True,
            )["best_config"].item()
            best_config = model_selection(
                data_file=val_stream,
                base_config=base_config,
                epochs=1,
                max_evals=max_evals,
                device=device,
            )
            np.savez_compressed(
                os.path.join(config_path, f"config_cwa_with_base_config_{stream_name}"),
                best_config=best_config,
                base_config=base_config,
                seed=seed,
            )


if __name__ == "__main__":
    main()
