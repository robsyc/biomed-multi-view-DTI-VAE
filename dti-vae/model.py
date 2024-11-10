import re
from typing import List, Literal
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel
from bmfm_sm.core.data_modules.namespace import LateFusionStrategy

from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class BiomedMultiViewMoleculeEncoder(nn.Module):
    def __init__(
        self
        # hugging_face: bool = False
    ):
        super(BiomedMultiViewMoleculeEncoder, self).__init__()
        # Initialize the pretrained model
        # model = torch.load('../data_root/bmfm_model_dir/pretrained/MULTIVIEW_MODEL/biomed-smmv-with-coeff-agg.pth')
        biomed_smmv_pretrained = SmallMoleculeMultiViewModel.from_pretrained(
            LateFusionStrategy.ATTENTIONAL,
            model_path='../data_root/bmfm_model_dir/biomed-smmv-base.pth',
            inference_mode=False,
        )
        # biomed_smmv_pretrained = SmallMoleculeMultiViewModel.from_pretrained(
        #     LateFusionStrategy.ATTENTIONAL,
        #     model_path='ibm/biomed.sm.mv-te-84m',
        #     hugging_face=True
        # )
        # Initialize the model subcomponents
        self.model_graph = biomed_smmv_pretrained.model_graph # output dim: 512
        self.model_image = biomed_smmv_pretrained.model_image # output dim: 512
        self.model_text = biomed_smmv_pretrained.model_text   # output dim: 768

    def forward(self, smiles):
        tokenized_smiles_list = []
        attention_mask_list = []
        image_tensors = []
        graph_emb = []

        for sm in smiles:
            # Prepare image and text data in batch format
            img_data = ImageFinetuneDataPipeline.smiles_to_image_format(sm)
            image_tensors.append(img_data['img'].squeeze(0)) # Remove extra batch dimension if present

            txt_data = TextFinetuneDataPipeline.smiles_to_text_format(sm)
            tokenized_smiles_list.append(txt_data['smiles.tokenized'].squeeze(0))
            attention_mask_list.append(txt_data['attention_mask'].squeeze(0))

            # Run the graph model on individual smiles
            graph_data = Graph2dFinetuneDataPipeline.smiles_to_graph_format(sm)
            graph_emb.append(self.model_graph(graph_data).squeeze(0))

        # Run the image and text models on the batched data
        image_batch = torch.stack(image_tensors, dim=0).to(device)
        tokenized_smiles_batch = pad_sequence(tokenized_smiles_list, batch_first=True).to(device)
        attention_mask_batch = pad_sequence(attention_mask_list, batch_first=True).to(device)

        image_emb = self.model_image(image_batch)
        text_emb = self.model_text(tokenized_smiles_batch, attention_mask_batch)

        # Stack the individually computed graph embeddings
        graph_emb = torch.stack(graph_emb, dim=0).to(device)

        return [graph_emb, image_emb, text_emb]

class T5ProstTargetEncoder(nn.Module):
    def __init__(self, verbose: bool = False):
        super(T5ProstTargetEncoder, self).__init__()
        self.verbose = verbose
        self.tokenizer = T5Tokenizer.from_pretrained('../data_root/ProstT5_model_dir', do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained("../data_root/ProstT5_model_dir").to(device)
        # tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False).to(device)
        # model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
        # only GPUs support half-precision (float16) currently; if you want to run on CPU use full-precision (float32) (not recommended, much slower)
        self.model.float() if device.type=='cpu' else self.model.half()
        if verbose:
            print(next(self.model.parameters()).device)
            print(self.model.dtype)

    def forward(self, sequences):
        if self.verbose:
            print("Encoding protein sequences")
            print("Number of sequences:", len(sequences))
            print("Max length sequence:", max(len(seq) for seq in sequences))

        # TODO: keep larger sequences on GPU!
        sequences = [sequence[:20] for sequence in sequences]
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))).upper() for sequence in sequences]
        sequences = ["<AA2fold> " + s for s in sequences]

        # Tokenize sequences
        ids = self.tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding="longest",
            return_tensors='pt'
        )
        ids = {key: tensor.to(device) for key, tensor in ids.items()}
        if self.verbose:
            print(ids.keys())
            print(ids['input_ids'].shape)
            print(ids['attention_mask'].shape)

        # Generate embeddings
        outputs = self.model(
            input_ids=ids['input_ids'],
            attention_mask=ids['attention_mask']
        ).last_hidden_state # (batch_size, seq_len, hidden_dim)
        if self.verbose:
            print("Output shape:", outputs.shape)
        outputs = outputs.mean(dim=1) # (batch_size, hidden_dim)
        if self.verbose:
            print("Mean-pooled output shape:", outputs.shape)
        return outputs

        # embeddings = []
        # for i in range(outputs.shape[0]):
        #     l = len(sequences[i])
        #     subseq = outputs[i, 1:l+1]
        #     embeddings.append(subseq.mean(dim=0))
        
        # return torch.stack(embeddings, dim=0)

class VAEAggregator(torch.nn.Module):
    def __init__(
            self, 
            input_dim_list: list, 
            hidden_dim: int = 512,
            mlp_layers: int = 2,
            latent_dim: int = 1024,
            verbose: bool = False
            ):
        super(VAEAggregator, self).__init__()
        # Basic initialization
        self.input_dim_list = input_dim_list
        self.min_input_dim = min(input_dim_list)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.verbose = verbose
        if verbose:
            print("""
Initializing MoleculeVAEAggregator with the following parameters:
input_dim_list: {}
hidden_dim: {}
mlp_layers: {}
latent_dim: {}""".format(input_dim_list, hidden_dim, mlp_layers, latent_dim))

        # Encoder (can handle multiple VAE-'views' e.g. graph, image, text)
        if len(self.input_dim_list) > 1:
        # project each input to the minimum input dimension
            self.E_input_projections = nn.ModuleList(
                [nn.Linear(dim, self.min_input_dim) for dim in input_dim_list]
            )
        # attention mechanism (in encoder)
            self.w_before_mean = nn.Sequential(
                nn.Linear(self.min_input_dim, self.min_input_dim),
                nn.Tanh(),
                nn.Linear(self.min_input_dim, 1, bias=False),
            )
        # project the concatenated input to the hidden dimension (with attention mechanism)
        self.E_hidden_projection = nn.Linear(self.min_input_dim * len(input_dim_list), hidden_dim)
        # hidden VAE MLP layers with hidden_dim
        E_hidden_layers = []
        for _ in range(mlp_layers):
            E_hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            E_hidden_layers.append(nn.LayerNorm(hidden_dim))
            E_hidden_layers.append(nn.SiLU())
            E_hidden_layers.append(nn.Dropout(0.1))
        self.E_hidden = nn.Sequential(*E_hidden_layers)
        # project to latent space (mu and logvar), ready for reparatemerization
        self.E_latent_projection = nn.Linear(hidden_dim, 2 * latent_dim) # mu and logvar

        # VAE elements
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()

        # Decoder
        # project from latent space to hidden dimension
        self.D_latent_projection = nn.Linear(latent_dim, hidden_dim)
        # hidden VAE MLP layers with hidden_dim
        D_hidden_layers = []
        for _ in range(mlp_layers):
            D_hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            D_hidden_layers.append(nn.LayerNorm(hidden_dim))
            D_hidden_layers.append(nn.SiLU())
            D_hidden_layers.append(nn.Dropout(0.1))
        self.D_hidden = nn.Sequential(*D_hidden_layers)
        # project to the concatenated input dimension
        D_output_projections = []
        for dim in input_dim_list:
            D_output_projections.append(nn.Linear(hidden_dim, dim))
        self.D_output_projections = nn.ModuleList(D_output_projections)

    def encode(self, x, eps: float = 1e-8):
        coeffs = None
        print("ENCODING") if self.verbose else None
        if len(self.input_dim_list) > 1:
            for tensor in x:
                print("Input tensor shape: ", tensor.shape) if self.verbose else None
            x = [encoder_projection(output) for encoder_projection, output in zip(self.E_input_projections, x)]
            for element in x:
                print("After input projection shape: ", element.shape) if self.verbose else None
            combined = torch.stack(x, dim=1)
            print("Combined shape: ", combined.shape) if self.verbose else None
            tmp = F.normalize(combined, dim=-1)
            w = self.w_before_mean(tmp).mean(0)
            print("W shape: ", w.shape) if self.verbose else None
            beta = F.softmax(w, dim=0)
            beta = beta.expand((combined.shape[0],) + beta.shape)
            print("Beta shape: ", beta.shape) if self.verbose else None
            logits = beta * combined
            coeffs = beta.squeeze(2)[0]
            print("Coeffs shape: ", coeffs.shape) if self.verbose else None
            x = torch.flatten(logits, start_dim=1)
        
        print("X shape: ", x.shape) if self.verbose else None
        encoded = self.E_hidden_projection(x)
        print("Encoded shape: ", encoded.shape) if self.verbose else None
        encoded = self.E_hidden(encoded)
        print("Encoded shape after hidden layers: ", encoded.shape) if self.verbose else None

        mu, logvar = torch.chunk(self.E_latent_projection(encoded), 2, dim=-1)
        print("Mu shape: ", mu.shape) if self.verbose else None
        print("Logvar shape: ", logvar.shape) if self.verbose else None
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril), coeffs
    
    def reparameterize(self, dist):
        print("\nREPAREMETERIZING\n") if self.verbose else None
        # implementation from https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/
        return dist.rsample()
    
    def decode(self, z):
        print("DECODING") if self.verbose else None
        print("Latent z with shape: ", z.shape) if self.verbose else None
        decoded = self.D_latent_projection(z) # latent 2 hidden dim
        print("Decoded shape after projection: ", decoded.shape) if self.verbose else None
        decoded = self.D_hidden(decoded)      # MLP w/ hidden dim
        print("Decoded shape after hidden layers: ", decoded.shape) if self.verbose else None
        outputs = []
        for output_projection in self.D_output_projections:
            outputs.append(output_projection(decoded))
            print("Output shape: ", outputs[-1].shape) if self.verbose else None
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        dist, coeffs = self.encode(x)
        z = self.reparameterize(dist)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z, coeffs

class BranchVAE(nn.Module):
    """
    Single-branch VAE model for 
    - Creation of VAE embeddings; and
    - Reconstruction of embeddings
    """
    def __init__(
        self,
        base_model: Literal['smmv', 'prost'],
        hidden_dim: int,
        mlp_layers: int,
        latent_dim: int,
        verbose: bool = False
    ):
        super(BranchVAE, self).__init__()
        self.model_base = BiomedMultiViewMoleculeEncoder() if base_model == 'smmv' else T5ProstTargetEncoder()
        self.input_dim_list = [512, 512, 768] if base_model == 'smmv' else [1024]
        self.model_vae = VAEAggregator(
            input_dim_list = self.input_dim_list,
            hidden_dim = hidden_dim,
            mlp_layers = mlp_layers,
            latent_dim = latent_dim,
            verbose = verbose
        )
            
    def forward(self, x, return_coeffs: bool = False):
        emb = self.model_base(x)
        emb_reconstructed, z, coeffs = self.model_vae(emb)
        if return_coeffs:
            return emb, emb_reconstructed, z, coeffs
        return emb, emb_reconstructed, z

class MultiBranchVAE(nn.Module):
    """
    Multi-branch VAE model for drug-target interaction prediction.
    Inputs are of 2 types: 
        - DRUGS: SMILES strings; and 
        - TARGETS: Protein amino-acid sequences

    Each are encoded w/ BranchVAE models and
    dot product of the latent space embeddings is used for prediction.
    """
    def __init__(
        self,
        # hugging_face: bool = False,
        verbose: bool = False
    ):
        super(MultiBranchVAE, self).__init__()
        self.verbose = verbose
        self.model_drug = BranchVAE(
            base_model = 'smmv',
            hidden_dim = 512,
            mlp_layers = 2,
            latent_dim = 1024, # output is b x 1024
            verbose = verbose
        )
        self.model_target = BranchVAE(
            base_model = 'prost',
            hidden_dim = 512,
            mlp_layers = 2,
            latent_dim = 1024, # output is b x 1024
            verbose = verbose
        )

    def forward(self, smiles, sequences, return_coeffs: bool = True):
        emb_drug, emb_drug_reconstructed, z_drug, coeffs_drug = self.model_drug(smiles, return_coeffs)
        emb_target, emb_target_reconstructed, z_target, coeffs_target = self.model_target(sequences, return_coeffs)
        y = torch.matmul(z_drug, z_target.T)
        return y, emb_drug, emb_target, emb_drug_reconstructed, emb_target_reconstructed