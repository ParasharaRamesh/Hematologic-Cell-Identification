'''
This class contains the most generic code needed for all tasks.

The main train code needs hooks for working therefore any subclass needs to implement these hooks
'''
class Module:
    def __init__(self, name, dataset, save_dir):
        '''

        :param name: name of the experiement
        :param dataset: the dataset object which implements the method get_dataloaders()
        :param save_dir: path where the model weights can be saved
        '''
        self.name = name
        self.dataset = dataset
        self.save_dir = save_dir

    #TODO



