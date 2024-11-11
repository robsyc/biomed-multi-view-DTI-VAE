import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Tokenizer, T5EncoderModel
import re

from torch.nn.utils.rnn import pad_sequence
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel
from bmfm_sm.core.data_modules.namespace import LateFusionStrategy
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AA_SEQ_CAP = 20 # TODO: increase this on GPU

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
        sequences = [sequence[:AA_SEQ_CAP] for sequence in sequences]
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

        return graph_emb, image_emb, text_emb

class AttAggregator(torch.nn.Module):
    """Attentional Aggregator - see: https://arxiv.org/abs/2410.19704 and https://arxiv.org/abs/2209.15101
    Aggregates a set of embeddings: m views x batches x input_dim -> batches x input_dim
    """
    def __init__(
            self, 
            input_dim_list: list,
            hidden_dim: int = None,
            output_dim: int = None,
    ):
        super(AttAggregator, self).__init__()
        if hidden_dim is None:
            hidden_dim = min(input_dim_list)
        if output_dim is None:
            output_dim = min(input_dim_list)
        
        self.project_down = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for input_dim in input_dim_list
        ])
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.project_up = nn.Linear(hidden_dim * len(input_dim_list), output_dim)

    def forward(self, x_list):
        x_list = [project(x) for project, x in zip(self.project_down, x_list)]
        x = torch.stack(x_list, dim=1)
        att = self.attention(F.normalize(x, dim=-1)).mean(dim=0)
        att = F.softmax(att, dim=0)
        att = att.expand((x.shape[0],) + att.shape)
        coeffs = att.squeeze(2)[0]
        agg = att * x
        agg = torch.flatten(agg, start_dim=1)
        return self.project_up(agg), coeffs

class AttExpander(torch.nn.Module):
    """Attentional Generator - does the opposite of the Attentional Aggregator
    Generates a set of embeddings: batches x input_dim -> m views x batches x input_dim
    """
    def __init__(
            self, 
            input_dim: int,
            output_dim_list: list,
            hidden_dim: int = None,
    ):
        super(AttExpander, self).__init__()
        if hidden_dim is None:
            hidden_dim = min(output_dim_list)
        self.project_up = nn.Linear(input_dim, hidden_dim * len(output_dim_list))
        self.project_down = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for output_dim in output_dim_list
        ])

    def forward(self, x, coeffs=None):
        x = self.project_up(x)
        x = x.view(x.shape[0], len(self.project_down), -1)
        if coeffs is not None:
            att = coeffs.unsqueeze(1)
            att = att.expand((x.shape[0],) + att.shape)
            x = x * att

        x = [project(x[:, i]) for i, project in enumerate(self.project_down)]
        return x

class LatentVAEBlock(torch.nn.Module):
    """VAE block - see: https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/
    Encodes & decodes a fixed-length embedding: batches x input_dim -> batches x input_dim
    Generates a latent representation: batches x latent_dim
    """
    def __init__(
            self, 
            input_dim: int = 512, 
            hidden_dim: int = 512,
            mlp_layers: int = 2,
            latent_dim: int = 1024,
            dropout_prob: float = 0.1,
            **kwargs
        ):
        super(LatentVAEBlock, self).__init__()
        encoder_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_prob)
        ]
        for _ in range(mlp_layers):
            encoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_prob)
            ])
        encoder_layers.append(nn.Linear(hidden_dim, 2 * latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()

        decoder_layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_prob)
        ]
        for _ in range(mlp_layers):
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_prob)
            ])
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, eps: float = 1e-5):
        x = self.encoder(x)
        mu, log_var = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(log_var) + eps
        scale_tril = torch.diag_embed(scale)
        return torch.distributions.MultivariateNormal(mu, scale_tril)
    
    def reparameterize(self, dist):
        return dist.rsample()
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, compute_loss: bool = False):
        dist = self.encode(x)
        z = self.reparameterize(dist)
        x_hat = self.decode(z)
        if not compute_loss:
            return z, x_hat
        loss_recon = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl_divergence(dist, std_normal).mean()
        return z, x_hat, loss_recon, loss_kl

class MoleculeBranch(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int = 512,
            mlp_layers: int = 2,
            latent_dim: int = 1024,
            dropout_prob: float = 0.1,
            **kwargs
        ):
        super(MoleculeBranch, self).__init__()
        self.base_model = BiomedMultiViewMoleculeEncoder()
        self.aggregator = AttAggregator(
            input_dim_list=[512, 512, 768], # graph, image, text
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.vae = LatentVAEBlock(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            mlp_layers=mlp_layers,
            latent_dim=latent_dim,
            dropout_prob=dropout_prob,
        )
        self.generator = AttExpander(
            input_dim=hidden_dim,
            output_dim_list=[512, 512, 768], # graph, image, text
            hidden_dim=hidden_dim,
        )
        print(f"base_model: {get_model_params(self.base_model):,}")
        print(f"aggregator: {get_model_params(self.aggregator):,}")
        print(f"vae: {get_model_params(self.vae):,}")
        print(f"generator: {get_model_params(self.generator):,}")

    def forward(self, smiles, compute_loss: bool = False):
        graph_emb, image_emb, text_emb = self.base_model(smiles)
        agg, coeffs = self.aggregator([graph_emb, image_emb, text_emb])
        if not compute_loss:
            z, agg_hat = self.vae(agg)
            return z
        z, agg_hat, _, loss_kl = self.vae(agg, compute_loss=True)
        graph_emb_hat, image_emb_hat, text_emb_hat = self.generator(agg_hat, coeffs)
        loss_recon = torch.nn.functional.mse_loss(graph_emb_hat, graph_emb, reduction='mean')
        loss_recon += torch.nn.functional.mse_loss(image_emb_hat, image_emb, reduction='mean')
        loss_recon += torch.nn.functional.mse_loss(text_emb_hat, text_emb, reduction='mean')
        loss = loss_recon + loss_kl
        return z, coeffs, loss
    
class ProteinBranch(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int = 512,
            mlp_layers: int = 3,
            latent_dim: int = 1024,
            dropout_prob: float = 0.1,
            **kwargs
        ):
        super(ProteinBranch, self).__init__()
        self.base_model = T5ProstTargetEncoder()
        self.vae = LatentVAEBlock(
            input_dim = 1024,
            hidden_dim = hidden_dim,
            mlp_layers = mlp_layers,
            latent_dim = latent_dim,
            dropout_prob = dropout_prob,
        )
        print(f"base_model: {get_model_params(self.base_model):,}")
        print(f"vae: {get_model_params(self.vae):,}")

    def forward(self, sequences, compute_loss: bool = False):
        x = self.base_model(sequences)
        if not compute_loss:
            z, x_hat = self.vae(x)
            return z, x_hat
        z, x_hat, loss_recon, loss_kl = self.vae(x, compute_loss=True)
        loss = loss_recon + loss_kl
        return z, loss
        
def get_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)