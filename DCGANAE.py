import torch.nn as nn
import torch
from utils import sliced_wasserstein_distance,generalized_sliced_wasserstein_distance,distributional_sliced_wasserstein_distance,\
    distributional_generalized_sliced_wasserstein_distance,max_sliced_wasserstein_distance,cramer_loss
class LSUNEncoder(nn.Module):
    def __init__(self,image_size,latent_size,num_chanel,hidden_chanels=64):
        super(LSUNEncoder,self).__init__()
        self.image_size = image_size
        self.latent_size=latent_size
        self.num_chanel= num_chanel
        self.hidden_chanels = hidden_chanels
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_chanel, self.hidden_chanels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.hidden_chanels, self.hidden_chanels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.hidden_chanels * 2, self.hidden_chanels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.hidden_chanels * 4, self.hidden_chanels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.hidden_chanels * 8, self.latent_size, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        z=self.main(x).view(x.shape[0],-1)
        return z
class Discriminator(nn.Module):
    def __init__(self,image_size,latent_size,num_chanel,hidden_chanels=64):
        super(Discriminator,self).__init__()
        self.image_size = image_size
        self.latent_size=latent_size
        self.num_chanel= num_chanel
        self.hidden_chanels = hidden_chanels
        self.main1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_chanel, self.hidden_chanels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.hidden_chanels, self.hidden_chanels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.hidden_chanels * 2, self.hidden_chanels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.hidden_chanels * 4, self.hidden_chanels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 8),
            nn.Tanh()
        )
        self.main2= nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.hidden_chanels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.mainz= nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.latent_size,self.hidden_chanels * 8 ,1, stride=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.hidden_chanels * 8,self.hidden_chanels * 8 *4*4 ,1, stride=1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 8 *4*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_chanels * 8 *4*4*2,1),
            nn.Sigmoid()
        )
    def forward(self, x,z=None,flag=False):
        if(flag==False): 
            h=self.main1(x)
            y =self.main2(h).view(x.shape[0],-1)
        else:
            h =self.main1(x).view(x.shape[0],-1)
            h2= self.mainz(z.view(z.shape[0],self.latent_size,1,1)).view(z.shape[0],-1)
            y = self.fc(torch.cat([h,h2],dim=1))
        return y,h
class LSUNDecoder(nn.Module):
    def __init__(self,image_size,latent_size,num_chanel,hidden_chanels=64):
        super(LSUNDecoder,self).__init__()
        self.image_size = image_size
        self.latent_size=latent_size
        self.num_chanel= num_chanel
        self.hidden_chanels = hidden_chanels
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.latent_size, self.hidden_chanels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.hidden_chanels * 8, self.hidden_chanels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( self.hidden_chanels * 4, self.hidden_chanels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( self.hidden_chanels * 2, self.hidden_chanels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( self.hidden_chanels, self.num_chanel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        x= self.main(z.view(z.shape[0],self.latent_size,1,1))
        return x


class DCGANAE(nn.Module):
    def __init__(self,image_size,latent_size,num_chanel,hidden_chanels,device):
        super(DCGANAE, self).__init__()
        self.image_size = image_size
        self.num_chanel = num_chanel
        self.latent_size = latent_size
        self.hidden_chanels=hidden_chanels
        self.device = device
        self.encoder=LSUNEncoder(image_size,latent_size,num_chanel,hidden_chanels)
        self.decoder = LSUNDecoder(image_size,latent_size,num_chanel,hidden_chanels)

    def compute_loss_SWD(self, discriminator, optimizer, minibatch, rand_dist, num_projection, p=2):
        label = torch.full((minibatch.shape[0],), 1, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, data = discriminator(data)
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward(retain_graph=True)
        optimizer.step()
        y_fake, data_fake = discriminator(data_fake)
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward(retain_graph=True)
        optimizer.step()
        _swd = sliced_wasserstein_distance(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1),
                                           num_projection, p,
                                           self.device)

        return _swd

    def compute_loss_GSWD(self, discriminator, optimizer, minibatch, rand_dist, g, r, num_projection, p=2):
        label = torch.full((minibatch.shape[0],), 1, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, data = discriminator(data)
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward(retain_graph=True)
        optimizer.step()
        y_fake, data_fake = discriminator(data_fake)
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward(retain_graph=True)
        optimizer.step()
        _gswd = generalized_sliced_wasserstein_distance(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1),
                                                        g, r,
                                                        num_projection, p,
                                                        self.device)

        return _gswd

    def compute_lossDGSWD(self, discriminator, optimizer, minibatch, rand_dist, num_projections, tnet, op_tnet, g, r,
                          p=2, max_iter=100, lam=1):
        label = torch.full((minibatch.shape[0],), 1, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, data = discriminator(data)
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward(retain_graph=True)
        optimizer.step()
        y_fake, data_fake = discriminator(data_fake)
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward(retain_graph=True)
        optimizer.step()
        _dswd = distributional_generalized_sliced_wasserstein_distance(data.view(data.shape[0], -1),
                                                                       data_fake.view(data.shape[0], -1),
                                                                       num_projections, tnet, op_tnet, g, r,
                                                                       p, max_iter, lam,
                                                                       self.device)
        return _dswd

    def compute_loss_MSWD(self, discriminator, optimizer, minibatch, rand_dist, p=2, max_iter=100):
        label = torch.full((minibatch.shape[0],), 1, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, data = discriminator(data)
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward(retain_graph=True)
        optimizer.step()
        y_fake, data_fake = discriminator(data_fake)
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward(retain_graph=True)
        optimizer.step()
        _mswd = max_sliced_wasserstein_distance(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1),
                                                p, max_iter,
                                                self.device)
        return _mswd

    def compute_lossDSWD(self, discriminator, optimizer, minibatch, rand_dist, num_projections, tnet, op_tnet, p=2,
                         max_iter=100, lam=1):
        label = torch.full((minibatch.shape[0],), 1, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, data = discriminator(data)
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward(retain_graph=True)
        optimizer.step()
        y_fake, data_fake = discriminator(data_fake)
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward(retain_graph=True)
        optimizer.step()
        _dswd = distributional_sliced_wasserstein_distance(data.view(data.shape[0], -1),
                                                           data_fake.view(data.shape[0], -1), num_projections, tnet,
                                                           op_tnet,
                                                           p, max_iter, lam,
                                                           self.device)
        return _dswd

    def compute_loss_cramer(self, discriminator, optimizer, minibatch, rand_dist):
        label = torch.full((minibatch.shape[0],), 1, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, data = discriminator(data)
        errD_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD_real.backward(retain_graph=True)
        optimizer.step()
        y_fake, data_fake = discriminator(data_fake)
        label.fill_(0)
        errD_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD_fake.backward(retain_graph=True)
        optimizer.step()
        loss = cramer_loss(data.view(data.shape[0], -1), data_fake.view(data.shape[0], -1))
        return loss