import numpy as np
import math
import random
from collections import defaultdict
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, Input, Conv2D, Conv1D, Concatenate
import pandas as pd

puzzleQuantity = 64
windowSize = 51
gapSize = 0
inputCount = 15 

class DataGenerator(keras.utils.Sequence):
    def __init__(self, trainPartition, lenSize, species, windowSize, gapSize, inputCount, 
                 shuffleSet=True, batch_size=32):
        self.batch_size = batch_size
        self.trainPartition = trainPartition
        self.lenSize = lenSize
        self.shuffleSet = shuffleSet
        self.species = species
        self.indexes = np.arange(self.lenSize)
        self.inputCount = inputCount
        self.windowSize = windowSize
        self.gapSize = gapSize
    
    def dataRepositioner1D(self, permutationDataset, permutationIndex, proteinMatrix):
        """
        Generate repositioned protein data based on permutations.

        
        Parameters:
            permutationFileAddress (str): The file path or address to the permutation dataset file.
            permutationSize (int): The number of permutations to generate.
            proteinMatrix (numpy.ndarray): The protein data matrix from which to create the repositioned data.
            windowSize (int, optional): The size of the windows used for rearranging protein sequences. Default is 9.

        Returns:
            repositioned puzzle and the index for the permutations

        """
        #permutationDataset = np.load(permutationFileAddress)
        repositioned = []
        for windowIndex in range(self.inputCount):
            startingPosition = windowIndex*self.windowSize + random.randint(0,self.gapSize) #space in between the tiles
            repositioned.append(proteinMatrix[startingPosition:startingPosition+(self.windowSize-self.gapSize)])
        repositioned2d = []
        for item in permutationDataset[int(permutationIndex)]:
            repositioned2d.extend(repositioned[item])

        #repositioned=[repositioned[k] for k in permutationDataset[int(permutationIndex)]]
        return np.array(repositioned2d) #np.array(repositioned)
    
    def protToDict(self, proteinName):
        protDict =  defaultdict(dict)
        prot_file = open('dataset/embd/{}/{}.embd'.format(self.species, proteinName))
        embd_value = np.zeros((800,1024))
        oneHotEmbd_value = np.zeros((800,21))
        for index, prot_line in enumerate(prot_file):
            prot_aa, prot_line = prot_line.strip().split(':')
            embd_value[index] = np.array([float(x) for x in prot_line.split()])
            
        protDict[proteinName] = embd_value
        return protDict

    def __len__(self):
        'Denotes the number of batches per epoch'
        #print(self.lenSize,self.batch_size)
        return math.floor(self.lenSize / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.trainPartition[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        #print(np.array(XA), np.array(XB))
        return X, y #{"input_ens_1": XA, "input_ens_2": XB} #(XA, XB)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.lenSize)
        """if self.shuffle == True:
            np.random.shuffle(self.indexes)"""

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        XA = []
        XB = []
        XAz = []
        XBz = []
        y = []
        
        permutationDataset = np.load("../jigsaw/permutations_15indices_withCorrectOrder.npy")
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
    permutationIndices = [i for i in range(1,permutationSize)]
    random.shuffle(permutationIndices)
    permutationIndices.insert(0,0)
    permutationIndices2 = [i for i in range(1, permutationSize)]
    random.shuffle(permutationIndices2)
    permutationIndices2.insert(0,0)
    return np.array(permutationIndices), np.array(permutationIndices2)



def partitioning(datasetAddress, puzzleQuantity):
    dataset_file = open(datasetAddress, 'r')
    trainPartition = []
    df = {'proteinA':[], 'proteinB':[], 'label':[]}
    for line in dataset_file:
        protA,protB,label = line.strip().split('\t')
        permutationIndices, permutationIndices2 = repositioner1DNonRedundant(puzzleQuantity)
        df['proteinA'].append(protA)
        df['proteinB'].append(protB)
        df['label'].append(label)
        for idx in range(len(permutationIndices)):
            trainPartition.append('{}*{}*{}*{}*{}'.format(protA, protB, label, 
                                                       permutationIndices[idx], permutationIndices2[idx]))
    return trainPartition, pd.DataFrame(data=df)  
        

# Original train and val


def inputConvShared(input_layer, conv1, conv2, conv3, conv4):
    conv = conv1(input_layer)
    conv = conv2(conv)
    conv = conv3(conv)
    conv = conv4(conv)
    out = Flatten()(conv)
    return out

def pretrained_jigsaw(input_features):
    input_layers = []
    conv1 = Conv1D(32, 10, strides=2, activation='relu')
    conv2 = Conv1D(64, 5, strides=2, activation='relu')
    conv3 = Conv1D(96, 3, strides=1, activation='relu')
    conv4 = Conv1D(128, 3, strides=1, activation='relu')

    
    out = []
    for i in range(inputCount):
        input_layer = input_features[:,i]
        input_layers.append(input_layer)
        out.append(inputConvShared(input_layer, conv1, conv2, conv3, conv4))

    concatenated = Concatenate()(out)


    out = Dense(4096, activation='relu')(concatenated)
    out = Dropout(rate=0.3)(out)
    out = Dense(1024, activation='relu')(out)
    out = Dropout(rate=0.3)(out)
    out = Dense(256, activation='relu')(out)
    out = Dropout(rate=0.3)(out)
    output = Dense(puzzleQuantity, activation='softmax')(out)  

    jigsaw_model = keras.models.Model(inputs=input_features, outputs=output)
    jigsaw_model.load_weights("../jigsaw/models/t5_MultiCNN1D_64Puzzle_OneHot_15inputs_.h5")
    jigsaw_model.trainable = False  # Freeze the outer model
    return jigsaw_model



# MLP 

def ppi_model(input_features):
    #input_features = Input(shape=(800, 1024))
    convP1 = Conv1D(64, inputCount*(windowSize-gapSize), strides=1, activation='relu')(input_features)
    convP2 = Conv1D(64, 400, strides=200, activation='relu')(input_features)
    convP3 = Conv1D(64, 200, strides=100, activation='relu')(input_features)
    convP4 = Conv1D(64, 100, strides=50, activation='relu')(input_features)
    convP5 = Conv1D(64, 50, strides=25, activation='relu')(input_features)
    convP6 = Conv1D(64, 20, strides=10, activation='relu')(input_features)

    out1 = Flatten()(convP1)
    out2 = Flatten()(convP2)
    out3 = Flatten()(convP3)
    out4 = Flatten()(convP4)
    out5 = Flatten()(convP5)
    out6 = Flatten()(convP6)

    out1 = Dense(16, activation='relu')(out1)
    out1 = Dropout(rate=0.5)(out1)
    out2 = Dense(32, activation='relu')(out2)
    out2 = Dropout(rate=0.5)(out2)
    out3 = Dense(64, activation='relu')(out3)
    out3 = Dropout(rate=0.5)(out3)
    out4 = Dense(128, activation='relu')(out4)
    out4 = Dropout(rate=0.5)(out4)
    out5 = Dense(256, activation='relu')(out5)
    out5 = Dropout(rate=0.5)(out5)
    out6 = Dense(512, activation='relu')(out6)
    out6 = Dropout(rate=0.5)(out6)
    model = keras.models.Model(inputs=input_features, outputs=[out1, out2, out3, out4, out5, out6])
    return model



def ppi_model2D(input_features):
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




def predict(species, batch_size, savedModel1, savedModel2):
    testPartition, pairDf = partitioning("dataset/pairs/{}_test.tsv".format(species), 1)
    #test_generator = DataGenerator(testPartition, len(testPartition),  species, 
    #                               batch_size=batch_size, shuffleSet=False)
    
    #testPartition = partitioning("dataset/pairs/human_validation.tsv")
    test_generator = DataGenerator(testPartition, len(testPartition), species,
                                       windowSize=windowSize, gapSize=gapSize, inputCount=inputCount,
                                       batch_size=batch_size, shuffleSet=False)


    input_features = Input(shape=(inputCount*(windowSize-gapSize), 1024))
    input_features2 = Input(shape=(inputCount*(windowSize-gapSize), 1024))

    out1, out2, out3, out4, out5, out6 = ppi_model(input_features).output
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


    model = keras.models.Model(inputs=[input_features,input_features2], outputs=output)


    model.load_weights("models/{}.h5".format(savedModel1)) 
    predictionCNN1D = []
    y_pred_testing = model.predict(test_generator, batch_size=batch_size,verbose=1, workers=16).ravel()
    predictionCNN1D.extend(y_pred_testing)

    test_generator = DataGenerator(testPartition, len(testPartition), species,
                                       windowSize=windowSize, gapSize=gapSize, inputCount=inputCount,
                                       batch_size=batch_size, shuffleSet=False)

    input_features = Input(shape=(inputCount*(windowSize-gapSize), 1024))
    input_features2 = Input(shape=(inputCount*(windowSize-gapSize), 1024))

    out1, out2, out3, out4, out5, out6 = ppi_model2D(input_features).output
    out21, out22, out23, out24, out25, out26 = ppi_model2D(input_features2).output
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
    model = keras.models.Model(inputs=[input_features,input_features2], outputs=output)


    model.load_weights("models/{}.h5".format(savedModel2)) 
    predictionCNN2D = []
    y_pred_testing = model.predict(test_generator, batch_size=batch_size,verbose=1, workers=16).ravel()
    predictionCNN2D.extend(y_pred_testing)

    predictionENS = (np.array(predictionCNN1D)+np.array(predictionCNN2D))/2
    pairDf['prediction'] = predictionENS
    pairDf['predictionLabel'] = [round(i) for i in predictionENS] 


    pairDf.to_csv('results/{}_{}_{}.tsv'.format(species, savedModel1, savedModel2), index=False, sep='\t')


#predict('human', 111, 't5_MultiCNN1D_balanced_notValidation_jigsawFixWeight_merged', 't5_MultiCNN2D_balanced_notValidation_jigsawFixWeight_merged')

#predict('worm', 100, 't5_MultiCNN1D_balanced_notValidation_MultijigsawWithZero_PQ8_gp0_WS51', 't5_MultiCNN2D_balanced_notValidation_MultijigsawWithZero_PQ8_gp0_WS51')
predict('human', 111, 't5_MultiCNN1D_balanced_notValidation_MultijigsawWithZero_PQ8_gp0_WS51', 't5_MultiCNN2D_balanced_notValidation_MultijigsawWithZero_PQ8_gp0_WS51')

