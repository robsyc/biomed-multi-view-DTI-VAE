import re
from typing import List
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

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class BiomedMultiViewEncoder(nn.Module):
    def __init__(
        self
        # hugging_face: bool = False
    ):
        super(BiomedMultiViewEncoder, self).__init__()
        # model = torch.load('../data_root/bmfm_model_dir/pretrained/MULTIVIEW_MODEL/biomed-smmv-with-coeff-agg.pth')
        biomed_smmv_pretrained = SmallMoleculeMultiViewModel.from_pretrained(
            LateFusionStrategy.ATTENTIONAL,
            model_path='../data_root/bmfm_model_dir/pretrained/MULTIVIEW_MODEL/biomed-smmv-with-coeff-agg.pth',
            inference_mode=False,
        )
        # biomed_smmv_pretrained = SmallMoleculeMultiViewModel.from_pretrained(
        #     LateFusionStrategy.ATTENTIONAL,
        #     model_path='ibm/biomed.sm.mv-te-84m',
        #     hugging_face=True
        # )
        self.model_graph = biomed_smmv_pretrained.model_graph # output dim: 512
        self.model_image = biomed_smmv_pretrained.model_image # output dim: 512
        self.model_text = biomed_smmv_pretrained.model_text   # output dim: 768
        self.models = [self.model_graph, self.model_image, self.model_text]

    def forward(self, smiles: list):
        tokenized_smiles_list = []
        attention_mask_list = []
        image_tensors = []
        graph_emb = []

        for sm in smiles:
            img_data = ImageFinetuneDataPipeline.smiles_to_image_format(sm)
            image_tensors.append(img_data['img'].squeeze(0)) # Remove extra batch dimension if present

            txt_data = TextFinetuneDataPipeline.smiles_to_text_format(sm)
            tokenized_smiles_list.append(txt_data['smiles.tokenized'].squeeze(0))
            attention_mask_list.append(txt_data['attention_mask'].squeeze(0))

            graph_data = Graph2dFinetuneDataPipeline.smiles_to_graph_format(sm)
            graph_emb.append(self.model_graph(graph_data).squeeze(0))

        image_batch = torch.stack(image_tensors, dim=0).to(device)
        tokenized_smiles_batch = pad_sequence(tokenized_smiles_list, batch_first=True).to(device)
        attention_mask_batch = pad_sequence(attention_mask_list, batch_first=True).to(device)

        image_emb = self.model_image(image_batch)
        text_emb = self.model_text(tokenized_smiles_batch, attention_mask_batch)
        graph_emb = torch.stack(graph_emb, dim=0).to(device)

        return [graph_emb, image_emb, text_emb]

class MoleculeVAEAggregator(torch.nn.Module):
    def __init__(
            self, 
            input_dim_list=[512, 512, 768], 
            latent_dim=1024,
            mlp_layers=1,
            ):
        super(MoleculeVAEAggregator, self).__init__()
        # basic initialization
        self.input_dim_list = input_dim_list
        self.latent_dim = latent_dim
        self.min_dim_list = min(input_dim_list)

        # attention mechanism (in encoder)
        self.w_before_mean = nn.Sequential(
            nn.Linear(self.min_dim_list, self.min_dim_list),
            nn.Tanh(),
            nn.Linear(self.min_dim_list, 1, bias=False),
        )

        # encoder
        self.encoder_projections = nn.ModuleList(
            [nn.Linear(dim, self.min_dim_list) for dim in self.input_dim_list]
        )
        self.encoder_project = nn.Linear(self.min_dim_list * len(self.input_dim_list), latent_dim)
        encoder_layers = []
        for _ in range(mlp_layers):
            encoder_layers.append(nn.Linear(latent_dim, latent_dim))
            encoder_layers.append(nn.LayerNorm(latent_dim))
            encoder_layers.append(nn.SiLU())
            encoder_layers.append(nn.Dropout(0.1))
        encoder_layers.append(nn.Linear(latent_dim, 2 * latent_dim)) # mu and logvar
        self.encoder_mlp = nn.Sequential(*encoder_layers)

        # VAE elements
        self.softplus = nn.Softplus()
        self.silu = nn.SiLU()

        # decoder
        decoder_layers = [nn.Linear(latent_dim, latent_dim)]
        for _ in range(mlp_layers):
            decoder_layers.append(nn.Linear(latent_dim, latent_dim))
            decoder_layers.append(nn.LayerNorm(latent_dim))
            decoder_layers.append(nn.SiLU())
            decoder_layers.append(nn.Dropout(0.1))
        self.decoder_mlp = nn.Sequential(*decoder_layers)
        self.decoder_project = nn.Linear(latent_dim, self.min_dim_list * len(self.input_dim_list))
        self.decoder_projections = nn.ModuleList(
            [nn.Linear(self.min_dim_list, dim) for dim in self.input_dim_list]
        )

    def encode(self, x, eps: float = 1e-8):
        projections = [
            encoder_projection(output) for encoder_projection, output in zip(self.encoder_projections, x)
        ]
        combined = torch.stack(projections, dim=1)
        tmp = F.normalize(combined, dim=-1)
        w = self.w_before_mean(tmp).mean(0)
        beta = F.softmax(w, dim=0)
        beta = beta.expand((combined.shape[0],) + beta.shape)
        logits = beta * combined
        coeffs = beta.squeeze(2)[0]
        flat_logits = torch.flatten(logits, start_dim=1)
        encoded = self.encoder_project(flat_logits)
        encoded = self.encoder_mlp(encoded)

        # VAE
        mu, logvar = torch.chunk(encoded, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril), coeffs
    
    def reparameterize(self, dist):
        # implementation from https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/
        return dist.rsample()
    
    def decode(self, z):
        decoded = self.decoder_mlp(z)
        decoded = self.decoder_project(decoded)
        decoded = decoded.view(-1, len(self.input_dim_list), self.min_dim_list)
        out = []
        for i in range(len(self.input_dim_list)):
            out.append(self.decoder_projections[i](decoded[:, i]))
        return out
    
    def forward(self, x):
        dist, coeffs = self.encode(x)
        z = self.reparameterize(dist)
        decoded = self.decode(z)
        return decoded, z, coeffs
    

class T5TargetModel(nn.Module):
    def __init__(self):
        super(T5TargetModel, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained('../data_root/ProstT5_model_dir', do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained("../data_root/ProstT5_model_dir").to(device)
        # tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False).to(device)
        # model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
        self.model.float() if device.type=='cpu' else self.model.half()

    def forward(self, sequences: List[str]):
        # replace all rare/ambiguous amino acids by X and introduce white-space between all sequences
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))).upper() for sequence in sequences]
        # indicate the direction of the translation by prepending "<AA2fold>" if you go from 3Di to AAs (or if you want to embed AAs)
        sequences = ["<AA2fold>" + " " + s for s in sequences]
        # tokenize sequences
        ids = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest", return_tensors='pt').to(device)
        # generate embeddings
        outputs = self.model(ids.input_ids,  attention_mask=ids.attention_mask).last_hidden_state

        embeddings = []
        for i in range(outputs.shape[0]):
            l = len(sequences[i])
            subseq = outputs[i, 1:l+1]
            embeddings.append(subseq.mean(dim=0))
        
        return torch.stack(embeddings, dim=0)