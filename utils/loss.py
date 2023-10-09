from pytorch_msssim import SSIM

def ssim_loss(ssim_criterion, x,y):
    '''
    The Structural Similarity Index (SSIM) is a measure of similarity between two images, where a higher SSIM value indicates greater similarity. In the context of loss functions for optimization, you typically want to minimize a loss, so you need to convert the similarity into a loss value.

    :param ssim_criterion: SSIM() loss function imported from pytorch_mssim
    :param x:
    :param y:
    :return:
    '''
    assert isinstance(ssim_criterion, SSIM), "the criterion function passed is not an instance of SSIM() class"
    return 1 - ssim_criterion(x,y)