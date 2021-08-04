import logging

DATA_COLUMN_NAMES = {'Episode Number': 'EpisodeNumber',
                     'Total Rewards': 'TotalRewards',
                     'Average Loss': 'AverageLoss',
                     'Episode Length': 'EpisodeLength'}

class Logger(object):
    def __init__(self) -> None:
        super().__init__()

        self.logger = logging.getLogger(__name__)
        self.logger.info("\t".join(str(x) for x in DATA_COLUMN_NAMES.values()))
    
    def write(self, data: dict):
        """
            data = dict('episode number', 'total rewards',
            'average loss', 'episode length')
        """
        self.logger.info("\t".join(str(x) for x in data.values()))