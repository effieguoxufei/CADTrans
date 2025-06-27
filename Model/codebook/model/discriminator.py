from .layers.transformer import *
from .layers.improved_transformer import *
import torch.nn as nn
import torch
from config import *
from model.network import *
    

class SolidDiscriminator(nn.Module):
    """
    Discriminator for solid
    """

    def __init__(self):
        """
        Initializes model.
        """
        super(SolidDiscriminator, self).__init__()

        self.embed_dim = DISCRIMINATOR_CONFIG['embed_dim']
        self.logistic = nn.Linear(2**BIT, 32)
        self.param_fc = nn.Sequential(
            nn.Linear(32*SOLID_PARAM_SEQ, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(),
        )

        self.pos_embed = PositionalEncoding(max_len=MAX_SOLID, d_model=self.embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.LeakyReLU(),
        )

    def forward(self, param):
        """ forward pass """
        p_embeds = self.logistic(param).flatten(start_dim=2, end_dim=3)
        p_embeds = self.param_fc(p_embeds.flatten(0,1)).unflatten(0,(p_embeds.shape[0], p_embeds.shape[1]))  
        box_embeds = p_embeds
        disc_input = self.pos_embed(box_embeds.transpose(0,1))

        # Final classification
        classification = self.fc(disc_input)

        return classification


class ProfileDiscriminator(nn.Module):
    """
    Discriminator for profile data.
    """

    def __init__(self):
        """
        Initializes the model.
        """
        super(ProfileDiscriminator, self).__init__()

        self.embed_dim = DISCRIMINATOR_CONFIG['embed_dim']
        self.logistic = nn.Linear(2**BIT+PROFILE_PARAM_SEQ,32)
        self.bbox_fc = nn.Sequential(
            nn.Linear(32*PROFILE_PARAM_SEQ, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(),
        )
        self.pos_embed = PositionalEncoding(max_len=MAX_PROFILE, d_model=self.embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.LeakyReLU(),
        )

    def forward(self, coord):
        p_embeds = self.logistic(coord).flatten(start_dim=2, end_dim=3)
        coord_embed = self.bbox_fc(p_embeds.flatten(0,1)).unflatten(0,(p_embeds.shape[0], p_embeds.shape[1]))    
        encoder_input = self.pos_embed(coord_embed.transpose(0,1))

        """ Forward pass """
        classification = self.fc(encoder_input)


        return classification


class LoopDiscriminator(nn.Module):
    """
    Discriminator for loop data.
    """

    def __init__(self):
        """
        Initializes the model.
        """
        super(LoopDiscriminator, self).__init__()

        self.embed_dim = DISCRIMINATOR_CONFIG['embed_dim']
        self.logistic = nn.Linear(2**BIT+LOOP_PARAM_PAD,32)
        self.param_fc = nn.Sequential(
            nn.Linear(32*LOOP_PARAM_SEQ, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(),
        )
        self.pos_embed = PositionalEncoding(max_len=MAX_LOOP, d_model=self.embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.LeakyReLU(),
        )

    def forward(self, coord):
        p_embeds = self.logistic(coord).flatten(start_dim=2, end_dim=3)
        p_embeds = self.param_fc(p_embeds.flatten(0,1)).unflatten(0,(p_embeds.shape[0], p_embeds.shape[1]))   
        encoder_input = self.pos_embed(p_embeds.transpose(0,1))
        
        """ Forward pass """
        classification = self.fc(encoder_input)

        return classification