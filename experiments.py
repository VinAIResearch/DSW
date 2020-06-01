import torch
from torchvision.utils import save_image
def reconstruct(filename,input,encoder,decoder,image_size,num_chanel,device):
    with torch.no_grad():
        x_sample = input.to(device)
        x_reconstruct_mean = decoder(encoder(x_sample))

        save_image(torch.cat((x_sample, x_reconstruct_mean), dim=0).view(input.shape[0]*2, num_chanel, image_size, image_size),
                   filename)
def sampling(filename,fixednoise,decoder,num_sample,image_size,num_chanel):
    with torch.no_grad():

        sample = decoder(fixednoise)

        save_image(sample.view(num_sample, num_chanel, image_size, image_size), filename,scale_each=True,normalize=True)

def sampling_eps(filename,fixednoise,decoder,num_sample,image_size,num_chanel):
    with torch.no_grad():

        sample = decoder(fixednoise)

        save_image(sample.view(num_sample, num_chanel, image_size, image_size), filename+'_64.eps',scale_each=True,normalize=True,nrow=8)
        save_image(sample.view(num_sample, num_chanel, image_size, image_size)[:32], filename+'_32.eps',scale_each=True,normalize=True,nrow=8)
        save_image(sample.view(num_sample, num_chanel, image_size, image_size)[:16], filename+'_16.eps',scale_each=True,normalize=True,nrow=8)