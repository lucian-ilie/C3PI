import numpy as np
import math
import random
from collections import defaultdict
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, Input, Conv1D, Conv2D, Conv1D, Concatenate

puzzleQuantity = 8
windowSize = 51 # -> 12472466 # 50 -> 12472436
gapSize = 0
inputCount = 15 


class DataGenerator(keras.utils.Sequence):
    """
   Generates data for Keras
   """
    def __init__(self, trainPartition, lenSize,windowSize, gapSize, inputCount, shuffleSet=True, batch_size=64):
        """
       Initialization
       
       Args:
           trainPartition (list): List of training data partitions
           lenSize (int): Length of the training data
           windowSize (int): Size of the window for rearranging protein sequences
           gapSize (int): Size of the gap between windows
           inputCount (int): Number of input sequences
           shuffleSet (bool, optional): Whether to shuffle the dataset or not. Defaults to True.
           batch_size (int, optional): Batch size for data generation. Defaults to 64.
       """
        self.batch_size = batch_size
        self.trainPartition = trainPartition
        self.lenSize = lenSize
        self.shuffleSet = shuffleSet
        self.inputCount = inputCount
        self.windowSize = windowSize
        self.gapSize = gapSize
        self.on_epoch_end()
    
    def dataRepositioner1D(self, permutationDataset, permutationIndex, proteinMatrix):
        """
       Generate repositioned protein data based on permutations.

       Args:
           permutationDataset (numpy.ndarray): Dataset containing permutations.
           permutationIndex (int): Index of the permutation to be used.
           proteinMatrix (numpy.ndarray): Protein data matrix.

       Returns:
           numpy.ndarray: Repositioned protein data.
       """
        
        repositioned = []
        for windowIndex in range(self.inputCount):
            startingPosition = windowIndex*self.windowSize + random.randint(0,self.gapSize) #space in between the tiles
            repositioned.append(proteinMatrix[startingPosition:startingPosition+(self.windowSize-self.gapSize)])
        repositioned2d = []
        for item in permutationDataset[int(permutationIndex)]:
            repositioned2d.extend(repositioned[item])

        return np.array(repositioned2d) 
    
    def protToDict(self, proteinName):
        """
       Load protein embeddings from file and store them in a dictionary.

       Args:
           proteinName (str): Name of the protein.

       Returns:
           defaultdict: Dictionary containing protein embeddings.
       """
        protDict =  defaultdict(dict)
        prot_file = open('dataset/embd/human/{}.embd'.format(proteinName))
        embd_value = np.zeros((800,1024))
        for index, prot_line in enumerate(prot_file):
            prot_aa, prot_line = prot_line.strip().split(':')
            embd_value[index] = np.array([float(x) for x in prot_line.split()])
            
        protDict[proteinName] = embd_value
        return protDict

    def __len__(self):
        """
       Denotes the number of batches per epoch.

       Returns:
           int: Number of batches per epoch.
       """
        return math.floor(self.lenSize / self.batch_size)

    def __getitem__(self, index):
        """
       Generate one batch of data.

       Args:
           index (int): Index of the batch.

       Returns:
           tuple: A tuple containing the input data (X) and labels (y) for the batch.
       """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.trainPartition[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
       Updates indexes after each epoch.
       """
        self.indexes = np.arange(self.lenSize)
        
    def __data_generation(self, list_IDs_temp):
        """
       Generates data containing batch_size samples.

       Args:
           list_IDs_temp (list): List of IDs for the current batch.

       Returns:
           tuple: A tuple containing the input data (X) and labels (y) for the batch.
       """
        XA = []
        XB = []
        y = []
        
        permutationDataset = np.load("permutations.npy")
        protDictChunk = defaultdict(dict)
        for item in list_IDs_temp:
            protNameA, protNameB, protAALabel, permutationIndexA, permutationIndexB = item.split('*')
            if len(protDictChunk[protNameA]) == 0:
                protDict = self.protToDict(protNameA)
                protDictChunk.update(protDict)
                
            if len(protDictChunk[protNameB]) == 0:
                protDict = self.protToDict(protNameB)
                protDictChunk.update(protDict)
                
            repositionedA = []
            repositionedB = []
            
            repositionedA = self.dataRepositioner1D(permutationDataset, permutationIndexA, np.array(protDictChunk[protNameA]))
            repositionedB = self.dataRepositioner1D(permutationDataset, permutationIndexB, np.array(protDictChunk[protNameB]))
            
            
            
            XA.append(np.array(repositionedA))
            XB.append(np.array(repositionedB))
            y.append(int(protAALabel))

        if self.shuffleSet == True:    
            XIndex = np.arange(len(XA))
            random.shuffle(XIndex) 
            XAShuffled = [XA[k] for k in XIndex]
            XBShuffled = [XB[k] for k in XIndex]
            yShuffled = [y[k] for k in XIndex]
            return [np.array(XAShuffled), np.array(XBShuffled)], np.array(yShuffled)
        else:
            return [np.array(XA), np.array(XB)], np.array(y)




def repositioner1DNonRedundant(permutationSize):
    """
    Generate non-redundant permutation indices for repositioning.

    This function generates non-redundant permutation indices for rearranging protein sequences.
    The permutation indices are randomly shuffled and then inserted into two separate lists.
    Redundancy is avoided by excluding the index 0 from the shuffled lists.

    Parameters:
        permutationSize (int): The number of permutation indices to generate.

    Returns:
        numpy.ndarray: Two arrays of non-redundant permutation indices.
    """
    permutationIndices = [i for i in range(1,permutationSize)]
    random.shuffle(permutationIndices)
    permutationIndices.insert(0,0)
    permutationIndices2 = [i for i in range(1, permutationSize)]
    random.shuffle(permutationIndices2)
    permutationIndices2.insert(0,0)
    
    
    return np.array(permutationIndices), np.array(permutationIndices2)

def partitioning(datasetAddress, puzzleQuantity):
    """
    Partition the dataset into training data.

    This function reads a dataset file and partitions it into training data. Each line in the dataset
    file contains information about two proteins and a label. The function generates non-redundant
    permutation indices for each protein pair and creates training data entries by combining the
    protein names, label, and permutation indices.

    Parameters:
        datasetAddress (str): The file path of the dataset.
        puzzleQuantity (int): The number of puzzle permutations.

    Returns:
        list: A list containing training data entries.
    """
    dataset_file = open(datasetAddress, 'r')
    trainPartition = []
    for line in dataset_file:
        protA,protB,label = line.strip().split('\t')
        permutationIndices, permutationIndices2 = repositioner1DNonRedundant(puzzleQuantity)
        for idx in range(len(permutationIndices)):
            trainPartition.append('{}*{}*{}*{}*{}'.format(protA, protB, label, 
                                                       permutationIndices[idx], permutationIndices2[idx]))
    return trainPartition    

# Original train and val
trainPartition = partitioning("dataset/pairs/human_train_balanced.tsv", puzzleQuantity)
valPartition = partitioning("dataset/pairs/human_validationJigsaw.tsv", 1)

training_generator = DataGenerator(trainPartition, len(trainPartition),
                                   windowSize=windowSize, gapSize=gapSize, inputCount=inputCount)
validation_generator = DataGenerator(valPartition, len(valPartition),
                                     windowSize=windowSize, gapSize=gapSize, inputCount=inputCount, 
                                     shuffleSet=False, batch_size=19)





def ppi_model(input_features):
    """
    Create a protein-protein interaction prediction model.

    This function constructs a multi-output convolutional neural network model for predicting
    protein-protein interactions. The model consists of six convolutional layers followed by
    flattening, dense, and dropout layers for feature extraction and classification.

    Parameters:
        input_features (Tensor): Input tensor representing protein sequence features.

    Returns:
        keras.Model: A Keras model with six output layers, each corresponding to different
        feature representations extracted from the input protein sequences.
    """
    dimReP1 = Dense(inputCount*(windowSize-gapSize), activation='relu')(input_features)
    dimReP1 = Dropout(rate=0.5)(dimReP1)
    dimReP1 = Reshape((inputCount*(windowSize-gapSize),inputCount*(windowSize-gapSize),1))(dimReP1)

    convP11 = Conv2D(32, inputCount*(windowSize-gapSize), strides=1, activation='relu')(dimReP1)
    convP12 = Conv2D(32, 400, strides=200, activation='relu')(dimReP1)
    convP13 = Conv2D(32, 200, strides=100, activation='relu')(dimReP1)
    convP14 = Conv2D(32, 100, strides=50, activation='relu')(dimReP1)
    convP15 = Conv2D(32, 50, strides=25, activation='relu')(dimReP1)
    convP16 = Conv2D(32, 20, strides=10, activation='relu')(dimReP1)

    out11 = Flatten()(convP11)
    out12 = Flatten()(convP12)
    out13 = Flatten()(convP13)
    out14 = Flatten()(convP14)
    out15 = Flatten()(convP15)
    out16 = Flatten()(convP16)

    out11 = Dense(16, activation='relu')(out11)
    out11 = Dropout(rate=0.5)(out11)
    out12 = Dense(32, activation='relu')(out12)
    out12 = Dropout(rate=0.5)(out12)
    out13 = Dense(64, activation='relu')(out13)
    out13 = Dropout(rate=0.5)(out13)
    out14 = Dense(128, activation='relu')(out14)
    out14 = Dropout(rate=0.5)(out14)
    out15 = Dense(256, activation='relu')(out15)
    out15 = Dropout(rate=0.5)(out15)
    out16 = Dense(512, activation='relu')(out16)
    out16 = Dropout(rate=0.5)(out16)
    model = keras.models.Model(inputs=input_features, outputs=[out11, out12, out13, out14, out15, out16])
    return model

input_features1 = Input(shape=(inputCount*(windowSize-gapSize), 1024))
input_features2 = Input(shape=(inputCount*(windowSize-gapSize), 1024))

out1, out2, out3, out4, out5, out6 = ppi_model(input_features1).output
out21, out22, out23, out24, out25, out26 = ppi_model(input_features2).output
concatenated1 = Concatenate()([out1,out21])
concatenated2 = Concatenate()([out2,out22])
concatenated3 = Concatenate()([out3,out23])
concatenated4 = Concatenate()([out4,out24])
concatenated5 = Concatenate()([out5,out25])
concatenated6 = Concatenate()([out6,out26])


out1 = Dense(16, activation='relu')(concatenated1)
out1 = Dropout(rate=0.5)(out1)
out2 = Dense(32, activation='relu')(concatenated2)
out2 = Dropout(rate=0.5)(out2)
out3 = Dense(64, activation='relu')(concatenated3)
out3 = Dropout(rate=0.5)(out3)
out4 = Dense(128, activation='relu')(concatenated4)
out4 = Dropout(rate=0.5)(out4)
out5 = Dense(256, activation='relu')(concatenated5)
out5 = Dropout(rate=0.5)(out5)
out6 = Dense(512, activation='relu')(concatenated6)
out6 = Dropout(rate=0.5)(out6)

concatenated = Concatenate()([out1,out2,out3,out4,out5,out6])
out = Dense(64, activation='relu')(concatenated)
out = Dropout(rate=0.3)(out)
out = Dense(8, activation='relu')(out)
out = Dropout(rate=0.3)(out)
output = Dense(1, activation='sigmoid')(out)
model = keras.models.Model(inputs=[input_features1,input_features2], outputs=output)


aupr_metric = AUC(curve='PR')
optimizer_adam = keras.optimizers.Adam(learning_rate=(float)(0.001), beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=optimizer_adam, metrics=['acc', aupr_metric])
model.summary()



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint("models/CNN2D.h5", save_weights_only=True, monitor='val_loss',
                        mode='min', verbose=1, save_best_only=True)

model.fit(x=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    callbacks=[es, mc],
                    workers=16, verbose=2, epochs=100)
