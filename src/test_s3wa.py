"""Validation"""
import time
import os
import logging
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torch import optim
from s3wa import S3WA
import matplotlib.pyplot as plt


# dataset definition
class CSVDataset(Dataset):
    """Data"""

    # load the dataset
    def __init__(self, path, mask_path, device):
        # load the csv file as a dataframe
        data = np.genfromtxt(path, delimiter=",")
        mask = np.genfromtxt(mask_path, delimiter=",")


        # store the inputs and outputs
        self.x_train = data[:, :-1]
        self.y_train = data[:, -1]

        # ensure input data is floats
        self.x_train = torch.as_tensor(self.x_train).float().to(device)

        # Load the mask  
        self.mask = torch.from_numpy(mask).to(device)

        self.x_train = torch.cat((self.x_train, self.mask.unsqueeze(-1)), dim=-1)

        # label encode target and ensure the values are floats
        self.y_train = LabelEncoder().fit_transform(self.y_train)
        self.y_train = torch.as_tensor(self.y_train).to(device)
        self.shape = self.x_train.shape
        self.classes = torch.unique(self.y_train)
        self.n_classes = torch.numel(self.classes)

    # number of rows in the dataset
    def __len__(self):
        return len(self.x_train)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.x_train[idx], self.y_train[idx]]

    


def gmean_multiclass(preds_tensor, true_labels_tensor, classes_tensor):
    """prequential g-mean
    Returns a vector with gmean for each time step"""
    # Sensitivity of positive class = recall = true positive rate
    # Conventing to int and list to use dictionary keys


    classes = classes_tensor.int().tolist()
    preds = preds_tensor.int().tolist()
    true_labels = true_labels_tensor.int().tolist()
    preq_s_sens = {c: 0 for c in classes}
    preq_n_sens = {c: 0 for c in classes}
    running_sens = {c: 0 for c in classes}
    running_gmean = 0
    fading_factor = 0.999
    exp = 1 / len(classes)
    this_gmean_vector = []
    for y, f in zip(true_labels, preds):
        # Sensibility = true positive class rate
        test = int(f == y)  # test = true if correct prediction was made
        preq_s_sens[y] = test + fading_factor * preq_s_sens[y]
        preq_n_sens[y] = 1 + fading_factor * preq_n_sens[y]
        running_sens[y] = preq_s_sens[y] / preq_n_sens[y]
        # G-mean
        running_gmean = np.power(np.prod(list(running_sens.values())), exp)
        this_gmean_vector.append(running_gmean)
    return this_gmean_vector


def prequential_error_with_fading(predictions, true_labels, fading_factor=0.999):
    """Prequential error with fading factor for a batch of predictions"""
    preq_incorrect = 0
    preq_total = 0
    error_results = []

    for pred, true in zip(predictions, true_labels):
        preq_incorrect = fading_factor * preq_incorrect + int(pred != true)
        preq_total = fading_factor * preq_total + 1
        error = preq_incorrect / preq_total if preq_total != 0 else 0
        error_results.append(error)

    # Calculating accuracy from error
    accuracy_results = [1 - error for error in error_results]
    
    return accuracy_results
    

def plot_distance_epoch(distancias_por_epoca):
    """
    Plota as médias das distâncias entre os pesos do modelo Adam e S3WA por época.

    Args:
    - distancias_por_epoca: Lista contendo a média das distâncias de cada época.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(distancias_por_epoca, marker='o', linestyle='-', color='blue')
    plt.title('Média da Distância Entre os Pesos do S3WA e Adam por Época')
    plt.xlabel('Época')
    plt.ylabel('Média da Distância')
    plt.grid(True)
    plt.show()


def evaluate_model(csv_file=None, config=None, epochs=1, device=None, mask_file=None):
    """Test Model"""
    dataset = CSVDataset(csv_file, mask_file, device)   
    gmean_replicates = []
    preds_replicates = []
    true_labels_replicates = []
    mean_gmean_replicates = []
    mean_accuracy_replicates = []
    time_replicates = []
    best_gmean = 0
    best_replicate = []
    best_model = None
    distancias_medias_por_epoca = []
   
    prequential_accuracy_replicates = []
    
    train_dl = DataLoader(dataset, batch_size=200, shuffle=False)
    ck = 200
    chunks = []
    
    for i in range(1, len(dataset.x_train)):
          if len(chunks) < ck: 
           chunk = dataset[0:i]
        
          else:    
           chunk = dataset[i-ck:i]
          
          chunks.append(chunk)
    
    
    for epoch in range(epochs):
        logging.info("Starting epoch %s", epoch)
        # define the network
        
        model = S3WA(dataset.shape[1], int(dataset.shape[1]/2), dataset.n_classes, device, **config)  #int(dataset.shape[1]/2)
        tic = time.process_time()
        preds, true_labels = model.train_evaluate(train_dl)
        toc = time.process_time()

        elapsed_time = toc - tic
        
        # Calculating prequential accuracy
        preq_accuracy = prequential_error_with_fading(preds, true_labels)
        
        
        
        # collect results
        gmean_vector = gmean_multiclass(preds, true_labels, dataset.classes)
        gmean_replicates.append(gmean_vector)
        gmean = 100 * np.mean(gmean_vector)
        if gmean > best_gmean:
            best_gmean = gmean
            best_replicate = gmean_vector
            best_model = model
        mean_gmean_replicates.append(gmean)
        # Collecting results for prequential accuracy
        mean_preq_accuracy = np.mean(preq_accuracy) if preq_accuracy else 0
        mean_accuracy_replicates.append(mean_preq_accuracy)
        prequential_accuracy_replicates.append(preq_accuracy)
        preds_replicates.append(preds.cpu().numpy())
        true_labels_replicates.append(true_labels.cpu().numpy())
        time_replicates.append(elapsed_time)
    # returning best trained model and result matrices
        
        #distancias_medias_por_epoca.append(distancias)
    
    #plot_distance_epoch(distancias_medias_por_epoca)

    return (
        best_model,
        {
            "mean_gmean": mean_gmean_replicates,
            "gmean": gmean_replicates,
            "preds": preds_replicates,
            "true_labels": true_labels_replicates,
            "best_replicate": best_replicate,
            "time": time_replicates,
            "mean_preq_accuracy": mean_accuracy_replicates,
            "prequential_accuracy": prequential_accuracy_replicates,
        },
    )


def run(
    stream_type,
    stream_name,
    path,
    config_path,
    result_path,
    models_path,
    epochs,
    device,
):
    """Run experiment for a particular stream"""
    # Reproducibility
    seed = int(time.time())
    np.random.seed(seed)
    torch.manual_seed(seed)

    mask = [5 ,10, 20]

    for n in mask:

      if stream_type == "real":            
          # training stream
          test_stream = os.path.join(path, f"data/{stream_type}/train_{stream_name}.csv")
          mask_stream = os.path.join(path, f"data/{stream_type}/lab_train_{stream_name}_uni_l5.csv")
          if n == 10:
            mask_stream = mask_stream.replace("l5", "l10")
          elif n == 20:
            mask_stream = mask_stream.replace("l5", "l20")
              
      else:
          test_stream = os.path.join(path, f"data/{stream_type}/{stream_name}.csv")
          mask_stream = os.path.join(path, f"data/{stream_type}/lab_{stream_type}_{stream_name}_uni_l5.csv") 
          if n == 10:
            mask_stream = mask_stream.replace("l5", "l10")
          elif n == 20:
            mask_stream = mask_stream.replace("l5", "l20")  
      
      print(test_stream)
      # gathering hiperparamers from config files
      mlp_name = stream_name.replace("Sine1","Sine").replace("Sine2","Sine")
      mlp_name = mlp_name.replace("Agrawal1","Agrawal").replace("Agrawal2","Agrawal").replace("Agrawal3","Agrawal").replace("Agrawal4","Agrawal")
      mlp_name = mlp_name.replace("SEA1","SEA").replace("SEA2","SEA")
      mlp_name = mlp_name.replace("STAGGER1","STAGGER").replace("STAGGER2","STAGGER")
      
      best_config = np.load(
        os.path.join(config_path, f"config_cwa_with_base_config_{mlp_name}.npz"),
        allow_pickle=True,
      )["best_config"].item()
      try:
        base_config = np.load(
            os.path.join(config_path, f"config_mlp_{mlp_name}.npz"),
            allow_pickle=True,
        )["best_config"].item()
      except FileNotFoundError:
        base_config = {}
      best_config = {**best_config, **base_config}
      logging.info("Using best config: %s", best_config)
      best_model, result = evaluate_model(
        csv_file=test_stream, config=best_config, epochs=epochs, device=device, mask_file=mask_stream
      )
      print("best ", result["mean_gmean"])
      print("accuracy ", result["mean_preq_accuracy"])
      
      np.savez_compressed(
        os.path.join(result_path, f"s3wa_{stream_type}_{stream_name}_uni_l{n}"),
        best_config=best_config,
        seed=seed,
        **result,
      )
      #if best_model:
      #  torch.save(
      #      best_model.s3wa_model.state_dict(),
      #      os.path.join(models_path, f"model_s3wa_{stream_type}_{stream_name}_l{n}.pt"),
      #  )
      # Showing results
      logging.info("Finished S3WA | data stream %s", stream_name)
      


def read_options(default_streams):
    """Read command line options"""
    path = os.path.abspath("../")
    result_path = os.path.join(path, "results/")
    config_path = os.path.join(path, "configs/")
    models_path = os.path.join(path, "models/")
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "-r",
        "--real",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        default=default_streams["real"],  # default if nothing is provided
    )
    cli.add_argument(
        "-a",
        "--abrupt",
        nargs="*",
        default=default_streams["abrupt"],
    )
    cli.add_argument(
        "-w",
        "--gradual",
        nargs="*",
        default=default_streams["gradual"],
    )
    cli.add_argument(
        "-p",
        "--path",
        nargs="?",
        default=path,
    )
    cli.add_argument(
        "-c",
        "--config_path",
        nargs="?",
        default=config_path,
    )
    cli.add_argument(
        "-m",
        "--models_path",
        nargs="?",
        default=models_path,
    )
    cli.add_argument(
        "-o",
        "--result_path",
        nargs="?",
        default=result_path,
    )
    cli.add_argument(
        "-e",
        "--epochs",
        nargs="?",
        type=int,
        default=30,
    )
    args = cli.parse_args()
    return args


def main():
    """Calls experiments for all streams"""
    logging.basicConfig(
        encoding="utf-8",
        level=logging.INFO,
        format="[%(levelname)s %(asctime)s] %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S",
    )
    # result analysis setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data streams
    streams = {
        "real": [
            #"sensorClasses29and31",
            #"elec2",
            #"noaa",
            #"powersupplyDayNight",  
            #"chess",
            #"luxembourg",
            #"rialto",

        ],
        "abrupt": [
            #"Sine1",
            #"Sine2",
            #"Agrawal1",
            #"Agrawal2",
            #"Agrawal3",
            #"Agrawal4",
            #"SEA1", 
            #"SEA2",
            #"STAGGER1",
            #"STAGGER2",
        ],
        "gradual": [
            #"Sine1",
            #"Sine2",
            #"Agrawal1",
            #"Agrawal2",
            #"Agrawal3",
            #"Agrawal4",
            #"SEA1",
            #"SEA2",
            #"STAGGER1",
            "STAGGER2",
        ],
    }
    args = read_options(streams)
    for stream_name in args.real:
        logging.info("Submitting S3WA Real %s!", stream_name)
        run(
            "real",
            stream_name,
            args.path,
            args.config_path,
            args.result_path,
            args.models_path,
            args.epochs,
            device,
        )
    for stream_name in args.abrupt:
        logging.info("Submitting S3WA Abrupt %s!", stream_name)
        run(
            "abrupt",
            stream_name,
            args.path,
            args.config_path,
            args.result_path,
            args.models_path,
            args.epochs,
            device,
        )
    for stream_name in args.gradual:
        logging.info("Submitting CWA Gradual %s!", stream_name)
        run(
            "gradual",
            stream_name,
            args.path,
            args.config_path,
            args.result_path,
            args.models_path,
            args.epochs,
            device,
        )


if __name__ == "__main__":
    main()
