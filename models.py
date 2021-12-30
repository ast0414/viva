import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class MnistViVA(nn.Module):
    def __init__(self, x_dim=784, y_dim=10, z_dim=20, t_dim=2, hidden_dim=400, zeta=10.0, rho=0.5, y_prior=None, device='cpu'):
        super(MnistViVA, self).__init__()

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.t_dim = t_dim
        self.nu = self.t_dim - 1
        self.zeta = zeta
        self.rho = rho

        # ----------------------
        # p model -- Generative
        # ----------------------
        if y_prior is None:
            y_prior = 1 / y_dim * torch.ones(1, y_dim, device=device)
        else:
            y_prior = y_prior.to(device)

        # p(y)
        self.p_y = D.OneHotCategorical(probs=y_prior, validate_args=False)

        # p(t|y)
        self.decoder_t = nn.Sequential(
            nn.Linear(y_dim, 2 * t_dim),
        )

        # p(z|y,t)
        self.decoder_z = nn.Sequential(
            nn.Linear(y_dim + t_dim, 2 * z_dim),
        )

        # p(x|y,z,t)
        self.decoder_x = nn.Sequential(
            nn.Linear(y_dim + z_dim + t_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, x_dim)
        )

        # ----------------------
        # q model -- Inference
        # ----------------------

        # q(y|x) = Cat(y|pi_phi(x,t)) -- outputs parametrization of categorical distribution
        self.encoder_y = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, y_dim)
        )

        # q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x))) -- output parametrizations for mean and diagonal variance of a Normal distribution
        self.encoder_z = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * z_dim)
        )

        # q(t|x,y,z)
        self.encoder_t = nn.Sequential(
            nn.Linear(x_dim + y_dim + z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * self.t_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    # q(y|x) = Categorical(y|pi_phi(x))
    def encode_y(self, x):
        q_y_logit = self.encoder_y(x)
        q_y = D.OneHotCategorical(logits=q_y_logit)
        return q_y

    # q(z|x,y) = Normal(z|mu_phi(x,y), diag(sigma2_phi(x,y)))
    def encode_z(self, x, y):
        xy = torch.cat([x, y], dim=1)
        mu, var = self.encoder_z(xy).chunk(2, dim=-1)
        var = F.softplus(var) + 1e-5
        q_z_xy = D.Normal(mu, torch.sqrt(var), validate_args=False)

        return q_z_xy

    # q(t|x,y,z)
    def encode_t(self, x, y, z):
        xyz = torch.cat([x, y, z], dim=1)
        mu, var = self.encoder_t(xyz).chunk(2, dim=-1)
        var = F.softplus(var) + 1e-5
        q_t_xyz = D.Normal(mu, torch.sqrt(var), validate_args=False)

        return q_t_xyz

    # p(t|y) = Normal(t|mu_phi(y), diag(sigma2_phi(y)))
    def decode_t(self, y):
        mu, var = self.decoder_t(y).chunk(2, dim=-1)
        var = F.softplus(var) + 1e-5
        p_t_y = D.Normal(mu, torch.sqrt(var))
        return p_t_y

    # p(z|y,t) = Normal(z|mu_phi(y,t), diag(sigma2_phi(y,t)))
    def decode_z(self, y, t):
        yt = torch.cat([y, t], dim=1)
        mu, var = self.decoder_z(yt).chunk(2, dim=-1)
        var = F.softplus(var) + 1e-5
        p_z_yt = D.Normal(mu, torch.sqrt(var))
        return p_z_yt

    # p(x|y,z,t) = Bernoulli / Normal
    def decode_x(self, y, z, t):
        yzt = torch.cat([y, z, t], dim=1)
        p_x_logit = self.decoder_x(yzt)
        p_x_yzt = D.Bernoulli(logits=p_x_logit, validate_args=False)
        return p_x_yzt

    def tsne_loss(self, z_mu, z_std, t_logits, y_probs):
        batch_size = t_logits.size(0)

        zi_mu = z_mu.unsqueeze(1).expand(batch_size, batch_size, self.z_dim).reshape(batch_size ** 2, -1)
        zj_mu = z_mu.unsqueeze(0).expand(batch_size, batch_size, self.z_dim).reshape(batch_size ** 2, -1)

        zi_std = self.zeta * z_std
        zi_var = zi_std.pow(2).unsqueeze(1).expand(batch_size, batch_size, self.z_dim).reshape(batch_size ** 2, -1)

        # assuming a diagonal covariance matrix
        z_ij_std_euc_dist_sq = torch.sum((zi_mu - zj_mu).pow(2) / zi_var, dim=1)

        p_density_j_given_i = torch.exp(-0.5 * z_ij_std_euc_dist_sq)
        p_density_j_given_i = p_density_j_given_i.reshape(batch_size, batch_size)

        y_argmax = torch.argmax(y_probs, dim=1, keepdim=True)
        yi_argmax = y_argmax.unsqueeze(1).expand(batch_size, batch_size, 1).reshape(batch_size ** 2, -1)
        yj_argmax = y_argmax.unsqueeze(0).expand(batch_size, batch_size, 1).reshape(batch_size ** 2, -1)

        # reward/penalty for the same/different class prediction
        y_same = (yi_argmax == yj_argmax).float().reshape(batch_size, batch_size) * 2 - 1  # [-1, 1]
        scale = torch.ones_like(p_density_j_given_i) + self.rho * y_same
        p_density_j_given_i = p_density_j_given_i * scale

        same_sample_mask = torch.ones_like(p_density_j_given_i) - torch.eye(batch_size).to(p_density_j_given_i.device)
        p_density_j_given_i = p_density_j_given_i * same_sample_mask
        p_density_j_given_i_sum = torch.sum(p_density_j_given_i, dim=1, keepdim=True)
        divider = p_density_j_given_i_sum

        p_j_given_i = p_density_j_given_i / divider
        p_ij = (p_j_given_i + p_j_given_i.t()) / (2 * batch_size)

        # lower dimension
        ti = t_logits.unsqueeze(1).expand(batch_size, batch_size, self.t_dim).reshape(batch_size ** 2, -1)
        tj = t_logits.unsqueeze(0).expand(batch_size, batch_size, self.t_dim).reshape(batch_size ** 2, -1)

        t_ij_distance_sq = torch.sum((ti - tj).pow(2), dim=1)

        q_density_ij = torch.pow((t_ij_distance_sq / self.nu) + 1, -(self.nu + 1) / 2)
        q_density_ij = q_density_ij.reshape(batch_size, batch_size)
        q_density_ij = q_density_ij * same_sample_mask

        q_ij = q_density_ij / q_density_ij.sum()

        same_sample_mask = same_sample_mask.flatten().bool()
        masked_p = torch.masked_select(p_ij.flatten(), same_sample_mask)
        masked_q = torch.masked_select(q_ij.flatten(), same_sample_mask)

        kl = F.kl_div(torch.log(masked_q), masked_p, reduction='sum')

        return kl

    def forward(self, x, y=None, temperature=0.5):

        # Encode
        q_y_x = self.encode_y(x)

        if y is None:
            if self.training:
                y = F.gumbel_softmax(q_y_x.logits, tau=temperature, hard=False)
            else:
                argmax_y = torch.argmax(q_y_x.logits, dim=1)
                y = torch.zeros_like(q_y_x.logits)
                y[torch.arange(x.size(0)), argmax_y] = 1
            latent_loss = -q_y_x.entropy()
        else:
            if not self.training:
                raise AttributeError('y is given in eval mode!')
            latent_loss = torch.zeros_like(q_y_x.entropy())

        q_z_xy = self.encode_z(x, y)
        if self.training:
            z = q_z_xy.rsample()
        else:
            z = q_z_xy.mean

        q_t_xyz = self.encode_t(x, y, z)
        if self.training:
            t = q_t_xyz.rsample()
        else:
            t = q_t_xyz.mean

        # Decode
        p_t_y = self.decode_t(y)
        p_z_yt = self.decode_z(y, t)
        p_x_yzt = self.decode_x(y, z, t)

        recon_loss = -p_x_yzt.log_prob(x).sum(1)

        latent_loss = latent_loss\
                      - self.p_y.log_prob(y)\
                      - p_z_yt.log_prob(z).sum(1)\
                      + q_z_xy.log_prob(z).sum(1)\
                      - p_t_y.log_prob(t).sum(1)\
                      + q_t_xyz.log_prob(t).sum(1)

        tsne_loss = self.tsne_loss(q_z_xy.mean, q_z_xy.stddev, q_t_xyz.mean, q_y_x.probs)

        return recon_loss.sum(0), q_y_x.logits, q_t_xyz.mean, latent_loss.sum(0), tsne_loss
