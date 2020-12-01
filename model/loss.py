# coding = utf-8

from numpy import sqrt
from torch import mean, squeeze
from torch.nn import Module, PairwiseDistance
    
    
# TODO squeeze is not time-comusing. While it's still good to remove it
class ScaledL2Trans(Module):
    def __init__(self, original_dimension:int = 256, embedding_dimension: int = 16, to_scale: bool = False):
        super(ScaledL2Trans, self).__init__()

        self.__l2 = PairwiseDistance(p=2).cuda()
        self.__l1 = PairwiseDistance(p=1).cuda()

        if to_scale:
            self.__scale_factor_original = sqrt(original_dimension)
            self.__scale_factor_embedding = sqrt(embedding_dimension)
        else:
            self.__scale_factor_original = 1
            self.__scale_factor_embedding = 1



    def forward(self, database, query, db_embedding, query_embedding):
        original_l2 = self.__l2(squeeze(database), squeeze(query)) / self.__scale_factor_original
        embedding_l2 = self.__l2(squeeze(db_embedding), squeeze(query_embedding)) / self.__scale_factor_embedding 

        return self.__l1(original_l2.view([1, -1]), embedding_l2.view([1, -1]))[0] / database.shape[0]

    

class ScaledL2Recons(Module):
    def __init__(self, original_dimension: int = 256, to_scale: bool = False):
        super(ScaledL2Recons, self).__init__()

        self.__l2 = PairwiseDistance(p=2).cuda()

        if to_scale:
            self.__scale_factor = sqrt(original_dimension)
        else:
            self.__scale_factor = 1

    def forward(self, database, reconstructed):
        return mean(self.__l2(squeeze(database), squeeze(reconstructed))) / self.__scale_factor
