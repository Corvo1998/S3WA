"""Module Uniform S3WA"""
import torch
from torch import nn, optim
from torch.nn import init
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel



#-----------------------------------------------------------------------------------------------
class Autoencoder(nn.Module):
   def __init__(self, input_size, z_dim):
       super(Autoencoder, self).__init__()

       self.encoder = nn.Sequential(
           #entrada
           nn.Linear(input_size, 25),
           nn.ReLU(),
           nn.Linear(25, z_dim),
           
       )
       self.decoder = nn.Sequential(
           nn.Linear(z_dim, 25),
           nn.ReLU(),
           nn.Linear(25, input_size),
           
          
       )

   def add_noise(self,x):
        noise = torch.randn_like(x) * 0.1
        return x + noise

   def forward(self, x):
       x_enc = self.encoder(x)
       x_dec = self.decoder(x_enc)
       return x_dec


#-----------------------------------------------------------------------------------------------
class S3WA(nn.Module):
    """Constant Weight Average"""

    def __init__(
        self,
        n_inputs,
        z_dim,
        n_classes,
        device,
        *,
        start=100,
        alpha_zero=1,
        activation="relu",
        layers=10,
        lr=0.1,
        momentum=0.9,
        nesterovs_momentum=True,
        weight_decay=0.1,
        beta_1=0.9,
        beta_2=0.999,
        optimizer="adam",
    ):
        """define model elements"""
        super().__init__()
        self.start = start
        self.alpha_zero = alpha_zero
        self.activation = activation
        self.layers = layers
        self.lr = lr
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.optimizer = optimizer
        self.device = device
        self.out_act = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_dae = nn.MSELoss()
    
       
        self.autoencoder = Autoencoder(n_inputs-1,z_dim)
        # MLP
        self.model = self.build_layers(z_dim, n_classes)
        

        # S3WA
        self.s3wa_model = None
       
        
        

        # to gpu is available
        self.to(device)

    def weight_averaging(self, mu_old, w_model, n_model):
        """Update S3WA"""
        # Layer-wise call
        mu_new = mu_old + self.alpha_zero * (w_model - mu_old)
        return mu_new
            
    def build_layers(self, z_dim, n_classes):
        """Build architecture"""
        layer_sizes = [
            z_dim,
            *self.layers,
        ]
        # build layers
        layers_mlp = []
        for in_, out_ in zip(layer_sizes, layer_sizes[1:]):
            linear_layer_mlp = nn.Linear(in_, out_)
            # weight initialization
            if self.activation == "relu":
                init.kaiming_normal_(linear_layer_mlp.weight, nonlinearity="relu")
            else:
                init.xavier_normal_(linear_layer_mlp.weight)
            # adding layers to lists
            layers_mlp.append(linear_layer_mlp)
            # adding activations
            layers_mlp.append(self.select_activation())
        # Output layers
        layers_mlp.append(nn.Linear(layer_sizes[-1], n_classes))
        init.xavier_normal_(layers_mlp[-1].weight)
        return nn.Sequential(*layers_mlp)

    def select_activation(self):
        """Produces the activation function based on the choice"""
        if self.activation == "tanh":
            return nn.Tanh()
        if self.activation in ("sigmoid", "logistic"):
            return nn.Sigmoid()
        return nn.ReLU()

    def forward(self, X):
        """Forward propagate input"""
        X = self.model(X)
        return X
    

    def train_autoencoder(self, inputs, optimizer_dae):
        
        optimizer_dae.zero_grad()

        x_enc = self.autoencoder.add_noise(inputs.float())
        x_auto = self.autoencoder(x_enc)

        loss_dae = self.criterion_dae(x_auto, inputs.float())
        
        loss_dae.backward()
        optimizer_dae.step()

        return x_auto
    

    def select_optimizer(self):
        optimizer = optim.AdamW(
            self.parameters(),
            betas=(self.beta_1, self.beta_2),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
    
        return optimizer
    

    # train the model
    def train_evaluate(self, train_dl):
        """Test then train"""
        # define the optimization algorithm
        optimizer = self.select_optimizer()
        preds = torch.empty(0).to(self.device)
        true_labels = torch.empty(0).to(self.device)
        adam_weights_vectors = []
        s3wa_weights_vectors = []
        distancias = []
        # enumerate instances
        for time_step, (inputs, targets) in enumerate(train_dl):
            
            
            # clear the gradients, we clear them for each instance - strict online
            optimizer.zero_grad()
                    
            
            last_input = inputs[-1].unsqueeze(0)
            last_target = targets[-1].unsqueeze(0)
            
            # Use a indexação booleana com a máscara
            inputs_labeled = inputs[(inputs == 1).any(dim=1), :]
            targets_labeled = targets[(inputs == 1).any(dim=1)]

            inputs = inputs[:, :-1].float()
            inputs_labeled = inputs_labeled[:, :-1]
            last_input = last_input[:, :-1].float()
            
           
            ############################### Test #################################################
          
            ####### Autoencoder ########
            x_rec = self.autoencoder(last_input.float())


            x_enc = self.autoencoder.encoder(last_input.float())
            x_enc = x_enc.detach()
            

            ####### MLP ########

            yhat = self(x_enc)
            

            true_labels = torch.cat((true_labels, last_target))
            
            
            ############################### Train #################################################
            
            ####### Autoencoder ########
            train_dae = self.train_autoencoder(inputs, optimizer)
            
            
            if inputs_labeled.shape[0] > 0: 
             
            
             x_noisy = self.autoencoder.add_noise(inputs_labeled)
             out_enc = self.autoencoder.encoder(x_noisy.float())
             z_train = out_enc.clone().detach() 
             
             optimizer.zero_grad()

             train_yhat = self(z_train)
             # calculate loss
             loss = self.criterion(train_yhat, targets_labeled)
             # calculate gradient
             loss.backward()
             # update model weights
             optimizer.step()
             
            if time_step > self.start:
                if self.s3wa_model is None:
                    self.s3wa_model = AveragedModel(
                        self.model, avg_fn=self.weight_averaging
                    )
                
                yhat_averaged = self.out_act(self.s3wa_model(x_enc).detach())
                preds = torch.cat((preds, torch.argmax(yhat_averaged, axis=1)))
                
                self.s3wa_model.update_parameters(self.model)

                adam_weights_vector = torch.cat([p.data.view(-1) for p in self.model.parameters()])
                s3wa_weights_vector = torch.cat([p.data.view(-1) for p in self.s3wa_model.module.parameters()])

                # Adiciona os vetores de pesos às respectivas listas
                adam_weights_vectors.append(adam_weights_vector)
                s3wa_weights_vectors.append(s3wa_weights_vector)
            else:
                preds = torch.cat(
                    (preds, torch.argmax(self.out_act(yhat.detach()), axis=1))
                )


             
        
        distancia = [torch.sqrt(torch.sum((adam - s3wa) ** 2)).item() for adam, s3wa in zip(adam_weights_vectors, s3wa_weights_vectors)]
        #distancia = torch.sqrt(torch.sum((adam_weights_vector - s3wa_weights_vector) ** 2)) 
        
        distancias.append(distancia)
        
        
         
        return preds, true_labels,
