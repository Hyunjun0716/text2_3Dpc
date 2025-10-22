import torch
from torch.nn import Module

from .common import *
from .encoders import *
from .diffusion import *


class GaussianVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)

        # Text dimension from CLIP (512 for CLIP base model)
        text_dim = getattr(args, 'text_dim', 0)
        use_alignment_loss = getattr(args, 'use_alignment_loss', True)
        alignment_weight = getattr(args, 'alignment_weight', 0.1)

        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual, text_dim=text_dim),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            ),
            use_alignment_loss=use_alignment_loss,
            alignment_weight=alignment_weight
        )

    def get_loss(self, x, text_emb=None, writer=None, it=None, kl_weight=1.0):
        """
        Args:
            x:  Input point clouds, (B, N, d).
            text_emb: Text embeddings from CLIP. (B, text_dim). Optional.
        """
        batch_size, _, _ = x.size()
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        log_pz = standard_normal_logprob(z).sum(dim=1)  # (B, ), Independence assumption
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )
        loss_prior = (- log_pz - entropy).mean()

        loss_recons = self.diffusion.get_loss(x, z, text_emb=text_emb, writer=writer, it=it)

        loss = kl_weight * loss_prior + loss_recons

        if writer is not None:
            writer.add_scalar('train/loss_entropy', -entropy.mean(), it)
            writer.add_scalar('train/loss_prior', -log_pz.mean(), it)
            writer.add_scalar('train/loss_recons', loss_recons, it)

        return loss

    def sample(self, z, num_points, flexibility, text_emb=None, truncate_std=None):
        """
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
            text_emb: Text embeddings from CLIP. (B, text_dim). Optional.
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.sample(num_points, context=z, text_emb=text_emb, flexibility=flexibility)
        return samples
