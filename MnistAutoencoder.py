import torch.nn as nn
from utils import *
class G_function(nn.Module):
    def __init__(self, dim):
        super(G_function, self).__init__()
        self.dim=dim
        self.main1 = nn.Sequential(
            nn.Linear(self.dim, self.dim),
        )
        self.main2 = nn.Sequential(
            nn.Linear(self.dim, self.dim),
        )
        self.f = nn.ReLU()
    def forward(self,x,theta):
        theta= self.main2(theta)
        encoded_projections = torch.matmul(x, theta.transpose(0, 1))

        return self.f(encoded_projections)
class Encoder(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.image_size = image_size ** 2
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Linear(self.image_size, 4*hidden_size),
            nn.LeakyReLU(0.2,True),
            nn.Linear(4*hidden_size, 2 * hidden_size),
            nn.LeakyReLU(0.2,True),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LeakyReLU(0.2,True),
            nn.Linear(hidden_size, self.latent_size),

        )
    def forward(self,input):
        return self.main(input.view(-1,self.image_size))
class Decoder(nn.Module):
    def __init__(self,image_size,hidden_size,latent_size):
        super(Decoder,self).__init__()
        self.image_size = image_size**2
        self.hidden_size=hidden_size
        self.latent_size= latent_size
        self.main= nn.Sequential(
            nn.Linear(latent_size,hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(True),
            nn.Linear(2*hidden_size, 4 * hidden_size),
            nn.ReLU(True),
            nn.Linear(4*hidden_size,self.image_size),
            nn.ReLU(True)
        )
    def forward(self,input):
        return self.main(input)
class MnistAutoencoder(nn.Module):
    def __init__(self,image_size,hidden_size,latent_size,device):
        super(MnistAutoencoder, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.device = device
        self.encoder=Encoder(image_size,hidden_size,latent_size)
        self.decoder = Decoder(image_size,hidden_size,latent_size)
    def compute_loss_SWD(self,minibatch,rand_dist,num_projection,p=2):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0],self.latent_size)).to(self.device)
        data_fake= self.decoder(z_prior)

        _swd = sliced_wasserstein_distance(data.view(data.shape[0],-1),data_fake.view(data.shape[0],-1) ,
                                           num_projection, p,
                                           self.device)

        return _swd
    def compute_loss_GSWD(self,minibatch,rand_dist,g_function,r,num_projection,p=2):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0],self.latent_size)).to(self.device)
        data_fake= self.decoder(z_prior)

        _gswd = generalized_sliced_wasserstein_distance(data.view(data.shape[0],-1),data_fake.view(data.shape[0],-1) ,g_function,r,
                                           num_projection, p,
                                           self.device)

        return _gswd
    def compute_loss_JGSWD(self,minibatch,rand_dist,g_function,r,num_projection,p=2):
        data = minibatch.to(self.device)
        z_data= self.encoder(data)
        z_prior = rand_dist((data.shape[0],self.latent_size)).to(self.device)
        data_fake= self.decoder(z_prior)
        _gswd = generalized_sliced_wasserstein_distance(torch.cat([z_data,data.view(data.shape[0],-1)],dim=1)
                                           ,torch.cat([z_prior,data_fake.view(data.shape[0],-1)],dim=1) ,g_function,r,
                                           num_projection, p,
                                           self.device)

        return _gswd
    def compute_lossDGSWD(self,minibatch,rand_dist,num_projections,tnet,op_tnet,g,r,p=2,max_iter=100,lam=1):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        _dswd = distributional_generalized_sliced_wasserstein_distance(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1),num_projections,tnet,op_tnet,g,r,
                                                p, max_iter,lam,
                                                self.device)
        return _dswd

    def compute_lossJDGSWD(self, minibatch, rand_dist, num_projections, tnet, op_tnet,g,r, p=2, max_iter=100, lam=1):
        data = minibatch.to(self.device)
        z_data = self.encoder(data)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        _dswd = distributional_generalized_sliced_wasserstein_distance(torch.cat([z_data, data.view(data.shape[0], -1)], dim=1)
                                                           , torch.cat([z_prior, data_fake.view(data.shape[0], -1)],
                                                                       dim=1),
                                                           num_projections, tnet, op_tnet,g,r,
                                                           p, max_iter, lam,
                                                           self.device)
        return _dswd
    def compute_loss_JSWD(self,minibatch,rand_dist,num_projection,p=2):
        data = minibatch.to(self.device)
        z_data= self.encoder(data)
        z_prior = rand_dist((data.shape[0],self.latent_size)).to(self.device)
        data_fake= self.decoder(z_prior)
        _swd = sliced_wasserstein_distance(torch.cat([z_data,data.view(data.shape[0],-1)],dim=1)
                                           ,torch.cat([z_prior,data_fake.view(data.shape[0],-1)],dim=1) ,
                                           num_projection, p,
                                           self.device)

        return _swd
    def compute_lossJDSWD(self,minibatch,rand_dist,num_projections,tnet,op_tnet,p=2,max_iter=100,lam=1):
        data = minibatch.to(self.device)
        z_data= self.encoder(data)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        _dswd = distributional_sliced_wasserstein_distance(torch.cat([z_data,data.view(data.shape[0],-1)],dim=1)
                                           ,torch.cat([z_prior,data_fake.view(data.shape[0],-1)],dim=1) ,
                                                           num_projections,tnet,op_tnet,
                                                p, max_iter,lam,
                                                self.device)
        return _dswd
    def compute_loss_MSWD(self,minibatch,rand_dist,p=2,max_iter=100):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        _mswd = max_sliced_wasserstein_distance(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1),
                                            p,max_iter,
                                           self.device)
        return _mswd
    def compute_loss_JMSWD(self,minibatch,rand_dist,p=2,max_iter=100):
        data = minibatch.to(self.device)
        z_data= self.encoder(data)
        z_prior = rand_dist((data.shape[0],self.latent_size)).to(self.device)
        data_fake= self.decoder(z_prior)
        _swd = max_sliced_wasserstein_distance(torch.cat([z_data,data.view(data.shape[0],-1)],dim=1)
                                           ,torch.cat([z_prior,data_fake.view(data.shape[0],-1)],dim=1) ,
                                            p,max_iter,
                                           self.device)

        return _swd
    def compute_lossDSWD(self,minibatch,rand_dist,num_projections,tnet,op_tnet,p=2,max_iter=100,lam=1):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        _dswd = distributional_sliced_wasserstein_distance(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1),num_projections,tnet,op_tnet,
                                                p, max_iter,lam,
                                                self.device)
        return _dswd

    def compute_loss_join_cramer(self,minibatch,rand_dist):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        x_prior = self.decoder(z_prior)
        z_encode = self.encoder(data)
        x1=torch.cat([z_encode,data.view(data.shape[0],-1)],dim=1)
        x2=torch.cat([z_prior,x_prior.view(data.shape[0],-1)],dim=1)
        loss = cramer_loss(x1,x2)
        return loss
    def compute_loss_cramer(self,minibatch,rand_dist):
        data=minibatch.to(self.device)
        z_prior=rand_dist((data.shape[0],self.latent_size)).to(self.device)
        x_prior=self.decoder(z_prior)
        loss = cramer_loss(data.view(data.shape[0],-1),x_prior.view(data.shape[0],-1))
        return loss
    def compute_join_wasserstein_vi_loss(self,minibatch,rand_dist,n_iter=100,p=2,e=0.1):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        x_prior = self.decoder(z_prior)
        z_encode = self.encoder(data)
        return compute_Sinkhorn_loss(torch.cat([z_encode,data.view(data.shape[0],-1)],dim=1),
                                           torch.cat([z_prior,x_prior.view(data.shape[0],-1)],dim=1),n_iter,p,e,self.device)
    def compute_wasserstein_vi_loss(self,minibatch,rand_dist,n_iter=100,p=2,e=0.1):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        x_prior = self.decoder(z_prior)
        return compute_Sinkhorn_loss(data.view(data.shape[0],-1),
                                           x_prior.view(data.shape[0],-1),n_iter,p,e,self.device)
    