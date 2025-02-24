import numpy as np 
from itertools import groupby, chain, starmap
from more_itertools import unzip


class StratifiedGroupKFold_custom:
    """
    Class Create a custom Kfold splitter of dataset: 
    Make sure to preserve proportion of class in train and test set. 
    Make sure that there distinct group in the train and test set except for class with single group instance.

    """
    def __init__(self, n_splits:int=5, shuffle:bool=True, random_state:int=42):
        """
        Initialise the class splitter based on the parameter of a traditional KFold 
            ----------
        Parameters: 
            n_splits(int): Number of fold desired
            shuffle(bool): Present only for compatibility purposes. Not used. 
            random_state(int): Initialise the random seed.
        """
        self.n_splits = n_splits
        self.random_state = random_state
        
    def split(self, X:any, y:np.array, groups:np.array): 
        """
        Split in train and test set with n_splits fold. 
            ----------
        Parameters: 
            X(pd.DataFrame): Features. Not used, here for compatibility purposes
            y(pd.Series): Class. Here moa
            groups(pd.Series): groups. How to group the data. here Metadata_InChIKey
            ----------
        Return:
            list(tuple): list of tuple with tuple[0] being train set and tuple[1] being the test set
                with a length of n_splits fold. 
        """
        #Fetch the index of the pd.Series (groups is used but X or y would have worked as well)
        index = np.arange(len(groups))

        #create two dict:
        #moa_group is a dict where keys a moa class and the values is a list of the unique different InChIKeys 
        #which ia part of this moa
        #moa_single is the same dict, but is composed of moa with a single InChIKey instead
        
        moa_group = {k: np.array(sorted(set(unzip(g)[1]))) 
                     for k, g in groupby(sorted(zip(y, groups), key=lambda x:x[0]), key=lambda x:x[0])}
        moa_single = {k: moa_group.pop(k)[0] for k in list(moa_group.keys()) if len(moa_group[k]) == 1}

        #For each moa composed of several InChIKey:
        #we select for every folds one InChIKey to put in the test set. We make sure that every 
        #InChIkey is used at least once in a test fold.
        rng = np.random.default_rng(self.random_state)
        select_moa_group = {k: np.hstack([rng.choice(np.arange(len(moa_group[k])), size=len(moa_group[k]), replace=False) 
                  for _ in range(int(np.ceil(self.n_splits / len(moa_group[k]))))])[:self.n_splits] 
                            for k in list(moa_group.keys())}

        #For every InChIKey, compute the result proportion of the moa put in the test set. Then average this for every 
        #class and then for every fold. The inverse of this number is the number of chunk that should be made 
        #from the list of indice for every moa_single
        numb = 1 / np.mean([np.mean(np.hstack(
            [index[groups == moa_group[k][select_moa_group[k]][n]].shape[0] / index[y == k].shape[0] 
                                    for k in list(moa_group.keys())])) for n in range(self.n_splits)])

        # we round to the nearest int. In case we have .5 float, round will round to the nearest even int which 
        # is difficult to predict a consistent behaviour. Instead, we prefer to go at the disadvantage of the classifier
        # and put more into the test set. Then nearest int and in case of .5 nearest from 0. 
        n_split_moa_single = np.round(numb) + ((np.floor(numb) % 2 == 0) & ((numb - np.floor(numb)) == 0.5))

        #Fetch the indice for every moa put in the test set and stack them. Do that for every fold
        index_test_moa_group = [np.hstack([index[groups == moa_group[k][select_moa_group[k]][n]] 
                                           for k in list(moa_group.keys())]) for n in range(self.n_splits)]

        #Fetch the indice for every moa single then create n_split_moa_single chunk. Repeat the operation so 
        #n_split fold can be created. For every moa_single, a list of fold is created. 
        #these list using starmap to stack every array from these list.
        index_test_moa_single = list(starmap(lambda *x: np.hstack(x),
                     list(zip(*[list(chain(*[list(
                         np.array_split(rng.permutation(index[groups == moa_single[k]]), n_split_moa_single)) 
            for _ in range(int(np.ceil(self.n_splits / n_split_moa_single)))]))[:self.n_splits] 
                                for k in list(moa_single.keys())]))))

        #finally merge the index from moa_group and moa_single using starmap again to access the inside of 
        #both list which are ziped per fold
        test_fold = list(starmap(lambda *x: rng.permutation(np.hstack(x)), 
                                 list(zip(index_test_moa_group, index_test_moa_single))))
        #Once we have the index put for every fold in the test set, we take the complementary into the train set for every fold. 
        train_fold = [rng.permutation(index[~np.isin(index, test_fold[i])]) for i in range(self.n_splits)]
        
        return list(zip(train_fold, test_fold))