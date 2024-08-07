#!/usr/bin/env python
"""
    General useful utility functions for pytorch
    Date created: 30/8/19
    Python Version: 3.6
"""

import gputils.startup_guyga as gputils
import os, shutil, pickle, sys, random, itertools
from tqdm import tqdm
from time import localtime, strftime, time
import numpy as np, math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import torchvision.utils as torchvis_utils
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
import skimage.io as skio
from PIL import Image

identity = lambda x: x

__author__ = "Guy Gaziv"

def param_count_str(parameters):
    N = sum(p.numel() for p in parameters)
    if N > 1e6:
        return '%.2fM' % (N / 1000000.0)
    elif N > 1e3:
        return '%.1fK' % (N / 1000.0)
    else:
        return '%d' % N

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, relative=False):
        self.relative = relative
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.relative:
            if not self.count:
                self.scale = 100 / abs(val)
            val *= self.scale
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train_classifier(model, trainloader, testloader, optimizer, n_epochs=10, start_epoch=0, use_cuda=False, checkpoint_filepath=None, sum_writer=None, verbose=2, loss_select=True):
    criterion = nn.CrossEntropyLoss()
    best_acc = 0  # best test accuracy
    best_loss = np.inf
    global global_step
    global_step = 0
    if verbose == 1:
        pbar = tqdm(desc='Epochs', total=n_epochs)
        pbar.update(start_epoch)
    for epoch in range(start_epoch, n_epochs):
        if verbose > 1:
            print('\nEpoch: [%d | %d]' % (epoch + 1, n_epochs))
        train_loss, train_acc = train_test(trainloader, model, criterion, use_cuda, optimizer, sum_writer=sum_writer, verbose=verbose - 1)
        if testloader is not None:
            test_loss, test_acc = train_test(testloader, model, criterion, use_cuda, verbose=verbose - 1)
            if sum_writer:
                sum_writer.add_scalar('Test/Loss', test_loss, epoch)
                sum_writer.add_scalar('Test/Top1', test_acc, epoch)
            is_best = test_loss < best_loss if loss_select else test_acc > best_acc
            if is_best:
                best_acc = test_acc
                best_loss = test_loss
            # best_acc = max(test_acc, best_acc)
        else:
            is_best = train_loss < best_loss if loss_select else train_acc > best_acc
            if is_best:
                best_acc = train_acc
                best_loss = train_loss
        # # append logger file
        # logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        if verbose == 1:
            pbar.update()
        if checkpoint_filepath is not None:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, filepath=checkpoint_filepath)
    if verbose == 1:
        pbar.close()
    return best_acc

def train_test(loader, model, criterion, use_cuda, optimizer=None, sum_writer=None, verbose=1):
    global global_step
    if optimizer:
        # switch to train mode
        mode = 'Train'
        model.train()
    else:
        mode = 'Test '
        model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    if verbose > 0:
        bar = tqdm(desc=mode, total=len(loader))
    for batch_idx, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  #targets.cuda(async=True)
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data)
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            if sum_writer:
                sum_writer.add_scalar('Train/Loss', losses.val, global_step)
                sum_writer.add_scalar('Train/Top1', top1.val, global_step)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose > 0:  # plot progress
            bar.set_postfix_str('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        ))
            bar.update()
    if verbose > 0:
        bar.close()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best=False, checkpoint='checkpoint', filename='checkpoint.pth.tar', filepath=None, savebestonly=False):
    if filepath is None:
        filepath = os.path.join(checkpoint, filename)
    else:
        filename = os.path.basename(filepath).split('.')[0]
        checkpoint = os.path.dirname(filepath)
    if is_best or not savebestonly:
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, '%s_best.pth.tar' % filename))

class NormalizeBatch(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor.clone()
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if tensor.ndim == 4:
            tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        elif tensor.ndim == 3:
            tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        else:
            raise TypeError('tensor is not a torch image nor an image batch.')
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# class NormalizeInverse(transforms.Normalize):
class NormalizeInverse(NormalizeBatch):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor)

from PIL import ImageEnhance
transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)
class ImageJitter(object):
    def __init__(self, transformdict=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):

        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out

class NormalizeImageNet(transforms.Normalize):
    def __init__(self):
        super(NormalizeImageNet, self).__init__(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def tensor_transform(tensor, xfm):
    return torch.stack([xfm(x) for x in tensor])

from gputils.dataset_dumper import DumpedDataset

class CustomDataset(Dataset):
    def __init__(self, dataset, input_xfm=identity, output_xfm=identity):
        self.dataset = dataset
        self.input_xfm = input_xfm
        self.output_xfm = output_xfm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, output = self.dataset[index]
        return self.input_xfm(input), self.output_xfm(output)

class UnlabeledDataset(Dataset):
    def __init__(self, dataset, return_tup_index=0):
        self.dataset = dataset
        self.return_tup_index = return_tup_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][self.return_tup_index]

def sym_KL(logits_A, logits_B, dim=1):
    return .5 * (F.kl_div(F.log_softmax(logits_A, dim=dim), F.softmax(logits_B, dim=dim)) +
                 F.kl_div(F.log_softmax(logits_B, dim=dim), F.softmax(logits_A, dim=dim)))

def random_spatial_sample(tensor_NCHW, num1d_or_ratio):
    N, C, H, W = tensor_NCHW.shape
    if num1d_or_ratio < 1: # Ratio
        n = int(H * num1d_or_ratio)
    else:
        n = int(num1d_or_ratio)
    assert n > 0
    return tensor_NCHW.view(N, C, H*W)[..., sorted(np.random.permutation(H*W)[:n**2])].view(N, C, n, n)

def image_second_moment(gray_image):
    x, y = torch.tensor(np.mgrid[:gray_image.shape[0], :gray_image.shape[1]], dtype=torch.float32).cuda()
    m00 = gray_image.sum()
    m10 = (x * gray_image).sum()
    m01 = (y * gray_image).sum()
    # m02 = torch.sum((y - m01/m00)**2 * gray_image / m00)
    m02 = torch.sum((y - m01/m00)**2 * gray_image)
    # m20 = torch.sum((x - m10/m00)**2 * gray_image / m00)
    m20 = torch.sum((x - m10/m00)**2 * gray_image)
    return m02 + m20
    # mu22 = torch.sum((x - m01/m00)**2 * (y - m10/m00)**2 * gray_image)
    # return mu22

def extract_tensorboard_images(fpath, tag):
    import tensorflow as tf
    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)
    images = []
    with tf.InteractiveSession().as_default():
        for e in tf.train.summary_iterator(fpath):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    images.append(im)
    return images

def alpha_norm(input_matrix, alpha):
    """
        Converts matrix to vector then calculates the alpha norm
    """
    alpha_norm = ((input_matrix.view(-1))**alpha).mean()
    return alpha_norm

def total_variation_norm(input_matrix, beta):
    """
        Total variation norm is the second norm in the paper
        represented as R_V(x)
    """
    to_check = input_matrix[..., :-1, :-1]  # Trimmed: right - bottom
    one_bottom = input_matrix[..., 1:, :-1]  # Trimmed: top - right
    one_right = input_matrix[..., :-1, 1:]  # Trimmed: top - right
    total_variation = (((to_check - one_bottom)**2 +
                        (to_check - one_right)**2)**(beta/2)).mean()
    return total_variation

def euclidian_loss(org_matrix, target_matrix):
    """
        Euclidian loss is the main loss function in the paper
        ||fi(x) - fi(x_0)||_2^2& / ||fi(x_0)||_2^2
    """
    distance_matrix = target_matrix - org_matrix
    euclidian_distance = alpha_norm(distance_matrix, 2)
    normalized_euclidian_distance = euclidian_distance / alpha_norm(org_matrix, 2)
    return normalized_euclidian_distance

def pdist(x, y=None, vectorized=True):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    if vectorized:
        ind = torch.tril_indices(*dist.shape, offset=-1)
        dist = dist[ind[0], ind[1]]
    return torch.clamp(dist, 0.0, np.inf)

def hw_flatten(tensor):
    return tensor.view(*tensor.shape[:-2], -1)

def batch_flat(tensor):
    return tensor.view(len(tensor), -1)

def montager(dataset, n_max_images_class=40):
    images, labels = zip(*dataset)
    # print(type(images[0]))
    unique_labels, labels_counts = np.unique(labels, return_counts=True)
    n_images_class = labels_counts.min()
    k_sampled_images_class = min(n_images_class, n_max_images_class)
    # print(n_images_class, k_sampled_images_class)
    montage_images = list(itertools.chain(*[[images[index] for index in \
        random.sample(list(np.where(np.array(labels) == lbl)[0]), k_sampled_images_class)] for lbl in unique_labels]))
    # print(type(list(montage_images)))
    # print([type(x) for x in montage_images])
    # print([x.shape for x in montage_images])
    img = torchvis_utils.make_grid(montage_images, nrow=k_sampled_images_class)
    return img
    # return tensor.reshape(len(tensor), -1)

def sparsity_loss(x, alpha=1, k=3, avg=False):
    assert x.ndim == 1
    f = lambda z: (1 + alpha * z.abs()).log()
    for _ in range(k):
        x = f(x)
    if avg:
        return x.mean()
    else:
        return x.sum()

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # i = ctx.saved_variables[0]
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def make_cosine_sched(max_iter, min_mult=0.01, burn_in_iter=None, L=None):
    if burn_in_iter == None:
        burn_in_iter = .05 * max_iter
    if L == None:
        L = .2 * max_iter

    def lambda_sched(iter_i):
        if iter_i < burn_in_iter:
            return pow(iter_i / burn_in_iter, 4)
        elif iter_i > L:
            return min_mult + 0.5*(1 - min_mult)*(1 + math.cos(math.pi*(iter_i - L) / (max_iter - L)))
        else:
            return 1

    return lambda_sched


def benchmark_num_workers(dataset):
    for num_workers in range(2, mp.cpu_count(), 2):  
        train_loader = DataLoader(dataset, shuffle=True, 
                                  num_workers=num_workers, batch_size=64, 
                                  pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{%.1f} second, num_workers={}".format(end - start, num_workers))

    
def gram_matrix(feats):
    """ Following https://github.com/leongatys/PytorchNeuralStyleTransfer
    """
    b, c, h, w = feats.size()
    F = feats.view(b, c, h*w)
    G = torch.bmm(F, F.transpose(1,2))
    G.div_(h*w)
    return G


def gram_mse_loss(pred_feats, actual_feats, **kwargs):
    """ Following https://github.com/leongatys/PytorchNeuralStyleTransfer
    """
    return F.mse_loss(*map(gram_matrix, [pred_feats, actual_feats]), **kwargs)


def numpify(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return x
    

@torch.no_grad()
def get_activations(net, imgs_numpy_uint8, transform_norm=identity, device=torch.device('cuda:0')):
    """Compute activations for numpy NHWC uint8 tensor
    """
    if imgs_numpy_uint8.ndim < 4:
        imgs_numpy_uint8 = imgs_numpy_uint8[None]
    imgs = torch.tensor(imgs_numpy_uint8, device=device, requires_grad=False).permute(0, 3, 1, 2)
    imgs_torch = transform_norm(imgs.type(torch.cuda.FloatTensor) / 255.)
    imgs_act = net(imgs_torch).detach().squeeze().cpu().numpy()
    return imgs_act


class ImagepathsDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=identity):
        super().__init__()
        self.image_paths = image_paths
        self.transform = transform
    
    def __getitem__(self, index):
        img = skio.imread(self.image_paths[index])
        # img = np.array(Image.open(self.image_paths[index]).convert('RGB'))
        return self.transform(img)
    
    def __len__(self):
        return len(self.image_paths)
 
 
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=identity):
        super().__init__()
        self.images = images
        self.transform = transform
    
    def __getitem__(self, index):
        img = self.images[index]
        return self.transform(img)
    
    def __len__(self):
        return len(self.images)
 
 
@torch.no_grad()
def get_activations_from_images_batched(net, images, batch_size=100, n_threads=16, transform=None, transform_norm=identity,
                            dataloader_impl=True, device=torch.device('cuda:0'), do_tqdm=True):
    """Get activations for images. Compute in batches.
    """
    acts_list = []
    
    if dataloader_impl:
        if transform is None:
            my_transform = lambda img: transform_norm(torch.tensor(img, requires_grad=False).permute(2, 0, 1) / 255.)
        else:
            my_transform = transform
            
        dset = ImageDataset(images, my_transform)           
        it = DataLoader(dset, batch_size=batch_size, pin_memory=True, num_workers=n_threads)
        if do_tqdm: 
            it = tqdm(it)
        for img_batch in it:
            acts_list.append(net(img_batch.to(device)).detach().squeeze().cpu().numpy())
    else:
        it = gputils.chunks(images, batch_size)
        if do_tqdm: 
            it = tqdm(it)
        for imgs_numpy_uint8 in it:
            acts_list.append(get_activations(net, imgs_numpy_uint8))
    return np.vstack(acts_list)


@torch.no_grad()
def get_activations_from_paths_batched(net, image_paths, batch_size=100, n_threads=16, transform=None, transform_norm=identity,
                            dataloader_impl=True, device=torch.device('cuda:0'), do_tqdm=True):
    """Get activations for images. Compute in batches.
    """
    acts_list = []
    
    if dataloader_impl:
        if transform is None:
            my_transform = lambda img: transform_norm(torch.tensor(img, requires_grad=False).permute(2, 0, 1) / 255.)
        else:
            my_transform = transform
        dset = ImagepathsDataset(image_paths, my_transform)           
        it = DataLoader(dset, batch_size=batch_size, pin_memory=True, num_workers=n_threads)
        if do_tqdm: 
            it = tqdm(it)
        for img_batch in it:
            acts_list.append(net(img_batch.to(device)).detach().squeeze().cpu().numpy())
    else:
        with gputils.Pool(n_threads) as pool:
            it = gputils.chunks(image_paths, batch_size)
            if do_tqdm: 
                it = tqdm(it)
            for image_paths_batch in it:
                imgs_numpy_uint8 = np.array(pool.map(lambda image_path: skio.imread(image_path), image_paths_batch))
                acts_list.append(get_activations(net, imgs_numpy_uint8))
    return np.vstack(acts_list)

       
def get_activations_batched(net, images_or_paths, batch_size=100, n_threads=16, transform=None, transform_norm=identity,
                            dataloader_impl=True, device=torch.device('cuda:0'), do_tqdm=True):
    """Get activations for images. Compute in batches.
    """
    if isinstance(images_or_paths, np.ndarray):
        return get_activations_from_images_batched(net, images_or_paths, batch_size, 
                                                   n_threads, transform, transform_norm, dataloader_impl, 
                                                   device, do_tqdm=do_tqdm)
    else:
        return get_activations_from_paths_batched(net, images_or_paths, batch_size, 
                                                  n_threads, transform, transform_norm, dataloader_impl, 
                                                  device, do_tqdm=do_tqdm)


def pil2tor(pil_img, transform_norm=identity, device=None):
    tor_img = torch.tensor(pil_img, device=device, requires_grad=False)
    if pil_img.ndim == 4:
        tor_img = tor_img.permute(0, 3, 1, 2)
    elif pil_img.ndim == 3:
        tor_img = tor_img.permute(2, 0, 1).unsqueeze(0)
    else:
        raise NotImplementedError
    tor_img = transform_norm(tor_img.type(torch.cuda.FloatTensor) / 255.)
    return tor_img


def tor2pil(tor_img, inv_transform_norm=identity):
    tor_img = torch.round(255. * inv_transform_norm(tor_img).squeeze_(0))
    if tor_img.ndim == 4:
        tor_img = tor_img.permute(0, 2, 3, 1)
    elif tor_img.ndim == 3:
        tor_img = tor_img.permute(1, 2, 0)
    else:
        raise NotImplementedError
    pil_img = tor_img.detach().cpu().numpy().astype(np.uint8)
    return pil_img


def numpify(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return x


def torchify(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return x
    
    
def make_weights_for_balanced_classes(dset, nclasses):
    """Balanced Sampling between classes with torchvision DataLoader
    https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703   
    """                        
    count = [0] * nclasses                                                      
    for _, lbl in dset:                                                         
        count[lbl] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(dset)                                              
    for idx, (_, lbl) in enumerate(dset):                                          
        weight[idx] = weight_per_class[lbl]                                  
    return weight        


def get_dataloader_balance_sampler(dset, nclasses):
    """ For unbalanced dataset we create a weighted sampler
    Usage: 
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle = True,                              
                                                   sampler = sampler, num_workers=args.workers, pin_memory=True)   
    """
    weights = make_weights_for_balanced_classes(dset, nclasses)                                                                
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler 
    

def tempsigmoid(x, nd=3.0):
    temp = nd / torch.log(torch.tensor(9.0)) 
    return torch.sigmoid(x / temp) 


def set_seed(seed=0):
    """Similar to Pytorch-lightning's seed_evertyhing
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

if __name__ == '__main__':
    pass
    # # fpath = '/mnt/tmpfs/guyga/ssfmri2im/Sep19_21-27_alexnet_112_decay0005_fcmom50_momdrop_EncTrain/events.out.tfevents.1568917675.n99.mcl.weizmann.ac.il'
    # # tag = 'Train/EpochVoxRF'
    # fpath = '/mnt/tmpfs/guyga/ssfmri2im/enc/Nov24_18-26_vgg19mlsa16_112_decay0002_fcgl_chan32_batch512_epk150_corrwin5_EncTrain/events.out.tfevents.1574612778.n99.mcl.weizmann.ac.il'
    # tag = 'ValEnc/Vox_PWCorr_vs_Corr'
    # images = extract_tensorboard_images(fpath, tag)

    # from PIL import Image
    # images = list(map(Image.fromarray, images))
    # images[0].save('PWCorrVal.gif', save_all=True, append_images=images[1:], duration=60, loop=0)
    # print('guy')
