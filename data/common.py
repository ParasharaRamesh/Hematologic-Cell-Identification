from torchvision.datasets import ImageFolder


def get_dataset(inp_path, transformations=None):
    '''
    Given a folder with sub folders containing images, get all the images along with applying transformations

    :param inp_path:
    :param transformations:
    :return:
    '''
    return ImageFolder(root=inp_path, transform=transformations)
