import torch.nn as nn
import torch
from utils import sliced_wasserstein_distance,generalized_sliced_wasserstein_distance,distributional_sliced_wasserstein_distance,\
    distributional_generalized_sliced_wasserstein_distance,max_generalized_sliced_wasserstein_distance

class Encoder(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.image_size = image_size ** 2
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Linear(self.image_size, 4 * hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(4 * hidden_size, 2 * hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, self.latent_size),

        )

    def forward(self, input):
        return self.main(input.view(-1, self.image_size))


class Decoder(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size):
        super(Decoder, self).__init__()
        self.image_size = image_size ** 2
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(True),
            nn.Linear(2 * hidden_size, 4 * hidden_size),
            nn.ReLU(True),
            nn.Linear(4 * hidden_size, self.image_size),
            nn.ReLU(True)
        )

    def forward(self, input):
        return self.main(input)


class MnistAutoencoder(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size, device):
        super(MnistAutoencoder, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.device = device
        self.encoder = Encoder(image_size, hidden_size, latent_size)
        self.decoder = Decoder(image_size, hidden_size, latent_size)

    def compute_loss_SWD(self, minibatch, rand_dist, num_projection, p=2):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)

        _swd = sliced_wasserstein_distance(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1),
                                           num_projection, p,
                                           self.device)

        return _swd

    def compute_loss_MGSWNN(self, minibatch, rand_dist, gsw, max_iter, p=2):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        gswd = gsw.max_gsw(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1), iterations=max_iter)

        return gswd

    def compute_loss_JMGSWNN(self, minibatch, rand_dist, gsw, max_iter,p=2):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        x_prior = self.decoder(z_prior)
        z_encode = self.encoder(data)
        X = torch.cat([z_encode, data.view(minibatch.shape[0], -1)], dim=1)
        Y = torch.cat([z_prior, x_prior.view(minibatch.shape[0], -1)], dim=1)
        gswd = gsw.max_gsw(X, Y, iterations=max_iter)

        return gswd

    def compute_loss_GSWD(self, minibatch, rand_dist, g_function, r, num_projection, p=2):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)

        _gswd = generalized_sliced_wasserstein_distance(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1),
                                                        g_function, r,
                                                        num_projection, p,
                                                        self.device)

        return _gswd



    def compute_loss_JGSWD(self, minibatch, rand_dist, g_function, r, num_projection, p=2):
        data = minibatch.to(self.device)
        z_data = self.encoder(data)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        _gswd = generalized_sliced_wasserstein_distance(torch.cat([z_data, data.view(data.shape[0], -1)], dim=1)
                                                        ,
                                                        torch.cat([z_prior, data_fake.view(data.shape[0], -1)], dim=1),
                                                        g_function, r,
                                                        num_projection, p,
                                                        self.device)

        return _gswd

    def compute_lossDGSWD(self, minibatch, rand_dist, num_projections, tnet, op_tnet, g, r, p=2, max_iter=100, lam=1):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        _dswd = distributional_generalized_sliced_wasserstein_distance(data.view(data.shape[0], -1),
                                                                       data_fake.view(data.shape[0], -1),
                                                                       num_projections, tnet, op_tnet, g, r,
                                                                       p, max_iter, lam,
                                                                       self.device)
        return _dswd

    def compute_lossJDGSWD(self, minibatch, rand_dist, num_projections, tnet, op_tnet, g, r, p=2, max_iter=100, lam=1):
        data = minibatch.to(self.device)
        z_data = self.encoder(data)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        _dswd = distributional_generalized_sliced_wasserstein_distance(
            torch.cat([z_data, data.view(data.shape[0], -1)], dim=1)
            , torch.cat([z_prior, data_fake.view(data.shape[0], -1)],
                        dim=1),
            num_projections, tnet, op_tnet, g, r,
            p, max_iter, lam,
            self.device)
        return _dswd

    def compute_loss_JSWD(self, minibatch, rand_dist, num_projection, p=2):
        data = minibatch.to(self.device)
        z_data = self.encoder(data)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        _swd = sliced_wasserstein_distance(torch.cat([z_data, data.view(data.shape[0], -1)], dim=1)
                                           , torch.cat([z_prior, data_fake.view(data.shape[0], -1)], dim=1),
                                           num_projection, p,
                                           self.device)

        return _swd

    def compute_lossJDSWD(self, minibatch, rand_dist, num_projections, tnet, op_tnet, p=2, max_iter=100, lam=1):
        data = minibatch.to(self.device)
        z_data = self.encoder(data)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        _dswd = distributional_sliced_wasserstein_distance(torch.cat([z_data, data.view(data.shape[0], -1)], dim=1)
                                                           , torch.cat([z_prior, data_fake.view(data.shape[0], -1)],
                                                                       dim=1),
                                                           num_projections, tnet, op_tnet,
                                                           p, max_iter, lam,
                                                           self.device)
        return _dswd

    def compute_loss_MSWD(self, minibatch, rand_dist, gsw, max_iter):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        _mswd = gsw.max_gsw(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1), iterations=max_iter)
        return _mswd

    def compute_loss_MGSWD(self, minibatch, rand_dist, theta, theta_op, g, r, p=2, max_iter=100):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        _mswd = max_generalized_sliced_wasserstein_distance(data.view(data.shape[0], -1),
                                                            data_fake.view(data.shape[0], -1),
                                                            theta,
                                                            theta_op,
                                                            g, r,
                                                            p,
                                                            max_iter
                                                            )
        return _mswd

    def compute_loss_JMGSWD(self, minibatch, rand_dist, theta, theta_op, g, r, p=2, max_iter=100):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        x_prior = self.decoder(z_prior)
        z_encode = self.encoder(data)
        x1 = torch.cat([z_encode, data.view(data.shape[0], -1)], dim=1)
        x2 = torch.cat([z_prior, x_prior.view(data.shape[0], -1)], dim=1)
        _mswd = max_generalized_sliced_wasserstein_distance(x1,
                                                            x2,
                                                            theta,
                                                            theta_op,
                                                            g, r,
                                                            p,
                                                            max_iter
                                                            )
        return _mswd

    def compute_loss_JMSWD(self, minibatch, rand_dist, gsw,max_iter):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        x_prior = self.decoder(z_prior)
        z_encode = self.encoder(data)
        x1 = torch.cat([z_encode, data.view(data.shape[0], -1)], dim=1)
        x2 = torch.cat([z_prior, x_prior.view(data.shape[0], -1)], dim=1)
        _mswd = gsw.max_gsw(x1, x2, iterations=max_iter)
        return _mswd

        return _swd

    def compute_lossDSWD(self, minibatch, rand_dist, num_projections, tnet, op_tnet, p=2, max_iter=100, lam=1):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        _dswd = distributional_sliced_wasserstein_distance(data.view(data.shape[0], -1),
                                                           data_fake.view(data.shape[0], -1), num_projections, tnet,
                                                           op_tnet,
                                                           p, max_iter, lam,
                                                           self.device)
        return _dswd


