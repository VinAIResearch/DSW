
import numpy as np
import torch
from torch.autograd import Variable


def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections
def sliced_wasserstein_distance(first_samples,
                                second_samples,
                                num_projections=1000,
                                p=2,
                                device='cuda'):
    dim = second_samples.size(1)
    projections = rand_projections(dim, num_projections).to(device)
    first_projections = first_samples.matmul(projections.transpose(0, 1))
    second_projections = (second_samples.matmul(projections.transpose(0, 1)))
    wasserstein_distance = torch.abs((torch.sort(first_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(second_projections.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1),1./p)
    return torch.pow(torch.pow(wasserstein_distance, p).mean(),1./p)

def circular_function(x1, x2, theta, r, p):
    cost_matrix_1 = torch.sqrt(cost_matrix(x1, theta * r))
    cost_matrix_2 = torch.sqrt(cost_matrix(x2, theta * r))
    wasserstein_distance = torch.abs((torch.sort(cost_matrix_1.transpose(0, 1), dim=1)[0] -
                            torch.sort(cost_matrix_2.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.sqrt(torch.sum(torch.pow(wasserstein_distance, p), dim=1))
    return torch.pow(torch.pow(wasserstein_distance, p).mean(),1./p)


def generalized_sliced_wasserstein_distance(first_samples,
                                            second_samples,
                                            g_fuction, r=1,
                                            num_projections=1000,
                                            p=2,
                                            device='cuda'):
    embedding_dim = first_samples.size(1)
    projections = rand_projections(embedding_dim, num_projections).to(device)
    return g_fuction(first_samples, second_samples, projections, r, p)


def max_sliced_wasserstein_distance(first_samples,
                                    second_samples,
                                    p=2,
                                    max_iter=100,
                                    device='cuda'):
    theta = torch.randn((1, first_samples.shape[1]), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    opt = torch.optim.Adam([theta], lr=0.0001)
    for _ in range(max_iter):
        encoded_projections = torch.matmul(first_samples, theta.transpose(0, 1))
        distribution_projections = torch.matmul(second_samples, theta.transpose(0, 1))
        wasserstein_distance = torch.abs((torch.sort(encoded_projections)[0] -
                                torch.sort(distribution_projections)[0]))
        wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p))
        l = - wasserstein_distance
        opt.zero_grad()
        l.backward(retain_graph=True)
        opt.step()
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))

    return wasserstein_distance, theta


def distributional_generalized_sliced_wasserstein_distance(first_samples, second_samples, num_projections, f,
                                                           f_op, g_function, r,
                                                           p=2, max_iter=10, lam=1, device='cuda'):
    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    for _ in range(max_iter):
        projections = f(pro)
        reg = lam * cosine_distance_torch(projections, projections)
        wasserstein_distance = g_function(first_samples, second_samples, projections, r, p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()
    return wasserstein_distance


def distributional_sliced_wasserstein_distance(first_samples, second_samples, num_projections, f, f_op,
                                               p=2, max_iter=10, lam=1, device='cuda'):
    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    for _ in range(max_iter):
        projections = f(pro)
        reg = lam * cosine_distance_torch(projections, projections)
        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
        distribution_projections = (second_samples_detach.matmul(projections.transpose(0, 1)))
        wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                                torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
        wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1),1./p)
        wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(),1./p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()
    projections = f(pro)
    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    distribution_projections = (second_samples.matmul(projections.transpose(0, 1)))
    wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1. / p)
    wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)
    return wasserstein_distance


def train_net(embedding_dim, num_projections, tnet, tnet_op, max_iter=1000, device='cuda'):
    print('pretrain net')
    for _ in range(max_iter):
        # projections = tnet(rand_projections(embedding_dim, num_projections).to(device))
        projections_reg = tnet(rand_projections(embedding_dim, num_projections).to(device))
        reg = cosine_distance_torch(projections_reg, projections_reg)

        loss = reg

        tnet_op.zero_grad()
        loss.backward()
        tnet_op.step()
    print('DOne')


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))

def compute_Sinkhorn_loss(encoded_samples, distribution_samples, n_iter=100, p=2, e=0.1, device='cuda'):
    n = encoded_samples.size(0)
    m = distribution_samples.size(0)
    d = encoded_samples.size(1)

    x = encoded_samples.unsqueeze(1).expand(n, m, d)
    y = distribution_samples.unsqueeze(0).expand(n, m, d)
    C = torch.pow(torch.abs(x - y), p).sum(2)
    return sink_stabilized(C, e, n_iter, device=device)

def cost_matrix(encoded_smaples, distribution_samples, p=2):
    n = encoded_smaples.size(0)
    m = distribution_samples.size(0)
    d = encoded_smaples.size(1)
    x = encoded_smaples.unsqueeze(1).expand(n, m, d)
    y = distribution_samples.unsqueeze(0).expand(n, m, d)
    C = torch.pow(torch.abs(x - y), p).sum(2)
    return C
def phi_d(s, d):
    return torch.log((1 + 4 * s / (2 * d - 3))) * (-1. / 2)

def cramer_loss(x1, x2):
    m = x1.shape[0]
    d = x1.shape[1]
    gamma = (4 / (3 * m)) ** (2 / 5)
    loss = phi_d(cost_matrix(x1, x1, 2) / (4 * gamma), d).sum() + phi_d(cost_matrix(x2, x2, 2) / (4 * gamma),
                                                                        d).sum() - 2 * phi_d(
        cost_matrix(x1, x2, 2) / (4 * gamma), d).sum()

    loss = 1 / (2 * (m ** 2) * ((np.pi * gamma) ** (1. / 2))) * loss
    return loss

#implemented based on https://github.com/rythei/PyTorchOT
def sink_stabilized(M, reg, numItermax=1000, tau=1e2, stopThr=1e-9, warmstart=None, print_period=20, device='cuda'):
    a = Variable(torch.ones((M.size()[0], 1)) / M.size()[0]).to(device)
    b = Variable(torch.ones((M.size()[1], 1)) / M.size()[1]).to(device)

    # init data
    na = len(a)
    nb = len(b)

    cpt = 0
    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = Variable(torch.zeros((na, 1))).to(device), Variable(torch.zeros((nb, 1))).to(device)
    else:
        alpha, beta = warmstart

    u, v = Variable(torch.ones((na, 1)) / na).to(device), Variable(torch.ones((nb, 1)) / nb).to(device)

    def get_K(alpha, beta):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg)

    def get_Gamma(alpha, beta, u, v):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg + torch.log(u.view((na, 1))) + torch.log(
            v.view((1, nb))))

    # print(np.min(K))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:

        uprev = u
        vprev = v

        # sinkhorn update
        v = torch.div(b, (K.t().matmul(u) + 1e-16))
        u = torch.div(a, (K.matmul(v) + 1e-16))

        # remove numerical problems and store them in K
        if torch.max(torch.abs(u)).item() > tau or torch.max(torch.abs(v)).item() > tau:
            alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)
            u, v = Variable(torch.ones((na, 1)) / na).to(device), Variable(torch.ones((nb, 1)) / nb).to(device)

            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            transp = get_Gamma(alpha, beta, u, v)
            err = (torch.sum(transp) - b).norm(1).pow(2).item()

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        if np.any(np.isnan(u.data.cpu().numpy())) or np.any(np.isnan(v.data.cpu().numpy())):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break

        cpt += 1

    return torch.sum(get_Gamma(alpha, beta, u, v) * M)



