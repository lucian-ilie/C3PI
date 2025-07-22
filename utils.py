
import os.path
import os
import glob
def protABgenerator(dfAddress):
    """
    Generates a list of unique proteins from a pairs DataFrame.

    Parameters:
    - dfAddress (str): File path for the pairs DataFrame containing columns 'proteinA',
      'proteinB', and 'interact'.

    Returns:
    - list: A list of unique proteins extracted from the 'proteinA' and 'proteinB' columns.

    This function reads a pairs DataFrame from the specified file path, extracts unique
    proteins from the 'proteinA' and 'proteinB' columns, and returns a list containing these proteins.

    Example:
    >>> proteins_list = protABgenerator("/path/to/pairs_dataframe.tsv")
    >>> print(proteins_list)

    """
    pairDf = pd.read_csv(dfAddress, sep='\t', 
                        names=['proteinA','proteinB','interact'])
    protAB = list(set(pairDf.proteinA.values.tolist()+pairDf.proteinB.values.tolist()))
    return protAB

#print('ecoli', len(protABgenerator('dataset/pairs/ecoli_test.tsv')))
#print('fly', len(protABgenerator('dataset/pairs/fly_test.tsv')))
#print('mouse', len(protABgenerator('dataset/pairs/mouse_test.tsv')))
#print('yeast', len(protABgenerator('dataset/pairs/yeast_test.tsv')))
#print('worm', len(protABgenerator('dataset/pairs/worm_test.tsv')))
#print('human', len(protABgenerator('dataset/pairs/human_test.tsv')))

def protListGenerator(listAddress):
    """
    Reads a list of protein sequences from a file and calculates sequence statistics.

    Parameters:
    - listAddress (str): File path for the list containing protein identifiers and sequences.

    Returns:
    - tuple: A tuple containing a list of protein identifiers (converted to lowercase),
      maximum sequence length, and minimum sequence length.

    This function reads a list of protein sequences from the specified file path.
    It calculates and returns a tuple containing:
    1. A list of protein identifiers (converted to lowercase),
    2. The maximum length of the protein sequences,
    3. The minimum length of the protein sequences.

    Example:
    >>> sequence_list, max_length, min_length = readSequenceList("/path/to/protein_list.txt")
    >>> print(sequence_list)
    >>> print(max_length)
    >>> print(min_length)

    """
    f = open(listAddress, 'r')
    allPID = []
    maxLen = 0
    minLen = 300
    while True:
        pid = f.readline()
        pseq = f.readline()
        if pseq == "":
            break
        
        if len(pseq.strip())> maxLen:
            maxLen = len(pseq.strip())

        if len(pseq.strip())< minLen:
            minLen = len(pseq.strip())
        allPID.append(pid.strip()[1:].lower())
    return allPID, maxLen, minLen

def unUsedID (proteinList, proteins):
    for protein in proteins:
        try:
            proteinList.remove(protein.lower())
        except:
            pass
    
    return(proteinList)

def similarPairs(pairDf1Address, pairDf2Address):
    """
    Compares two pairs DataFrames based on protein interactions and returns common pairs.

    Parameters:
    - pairDf1Address (str): File path for the first pairs DataFrame containing columns 'proteinA',
      'proteinB', and 'interact'.
    - pairDf2Address (str): File path for the second pairs DataFrame containing columns 'proteinA',
      'proteinB', and 'interact'.

    Returns:
    - pd.DataFrame: A DataFrame containing common pairs of proteins and their interactions.

    This function reads two pairs DataFrames from the specified file paths, merges them based on the
    'proteinA' and 'proteinB' columns, and returns a DataFrame containing common pairs of proteins
    and their interactions.

    Example:
    >>> result_df = similarPairs("/path/to/pair_df1.tsv", "/path/to/pair_df2.tsv")
    >>> print(result_df)

    """
    pairTrainDf = pd.read_csv(pairDf1Address, sep='\t', 
                            names=['proteinA','proteinB','interact'])
    pairTestDf = pd.read_csv(pairDf2Address, sep='\t', 
                            names=['proteinA','proteinB','interact'])
    df = pairTrainDf.merge(pairTestDf, left_on=['proteinA','proteinB'], 
                           right_on=['proteinA','proteinB'])
    return df

def prunedFasta(unusedList, allPidAddress, outputAddress):
    """
    Prunes a FASTA file based on a list of unused identifiers and writes the result to a new file.

    Parameters:
    - unusedList (list): List of identifiers (excluding the leading '>') to be excluded from the output.
    - allPidAddress (str): The file path of the input FASTA file containing identifiers and sequences.
    - outputAddress (str): The file path for the output pruned FASTA file.

    This function reads a FASTA file containing identifiers and sequences, and writes to a new file
    excluding the sequences associated with identifiers present in the 'unusedList'. The resulting
    pruned FASTA file is saved at the specified 'outputAddress'.

    Example:
    >>> unused_identifiers = ['id1', 'id3']
    >>> prunedFasta(unused_identifiers, "/path/to/all_sequences.fasta", "/path/to/pruned_sequences.fasta")

    Note: The identifiers in 'unusedList' should be provided in lowercase for case-insensitive comparison.

    """
    f = open(allPidAddress, 'r')
    w = open(outputAddress, 'w')
    while True:
        pid = f.readline()
        pseq = f.readline()
        if pseq == "":
            break
        if pid.strip()[1:].lower() not in unusedList:
            w.write(pid)
            w.write(pseq)



#print(similarPairs('data/pairs/human_train.tsv', 'data/pairs/human_test.tsv'))

"""protAB = protABgenerator('dataset/pairs/{}_test.tsv'.format("human"))
w = open('dataset/seq/Pruned_human_test.fasta', 'w')
f = open('dataset/seq/Pruned_human.fasta')
while True:
    line = f.readline()
    if line =='':
        break
    if '>' in line:
        if line.strip()[1:] in protAB:
            w.write(line)
            w.write(f.readline())"""
"""
species = ['human', 'mouse', 'fly', 'yeast', 'ecoli', 'worm']
for organism in species: 
    allPID, maxLen, minLen = protListGenerator('data/seqs/{}.fasta'.format(organism))
    lengthAllPID = len(allPID)
    print('in {} max len of protein is:{} and min len is:{}'.format(
        organism,maxLen, minLen))
    
    protAB = protABgenerator('data/pairs/{}_test.tsv'.format(organism))
    if os.path.isfile('data/pairs/{}_train.tsv'.format(organism)):
        protAB.extend(protABgenerator('data/pairs/{}_train.tsv'.format(organism)))
        protAB = list(set(protAB))
    unusedList = unUsedID(allPID, protAB)

    print("{} orginal number of protein is: {}, number of unused protein: {}".format(
        organism,lengthAllPID,len(unusedList)))
    
    prunedFasta(unusedList,'data/seqs/{}.fasta'.format(organism), 
                'data/seqs/Pruned_{}.fasta'.format(organism))    
"""
def nameResolver(pairAddress):
    '''
    Check to see whether there is a same pair in Ecoli after removing duplicate sequences caused by naming problem  
    usage:
        nameResolver("dataset/pairs/ecoli_test.tsv")
    '''
    pairDf = pd.read_csv(pairAddress, sep='\t', 
                        names=['proteinA','proteinB','interact'])
    print(pairDf.shape)
    pairDf = pairDf.drop_duplicates(subset=['proteinA','proteinB'])
    print(pairDf.shape)
    print(pairDf)
    
def removeDeficientEmbedding(fastaAddress, embdAddress, expectedEmbdLength):
    '''
    Remove deficient embeddings caused by time limits in sbatch.

    Parameters:
    - fastaAddress (str): The file path to the input FASTA file containing protein sequences.
    - embdAddress (str): The directory path where the embeddings are stored.
    - expectedEmbdLength (int): The expected length of each embedding.

    Returns:
    None

    This function reads protein sequences from the specified FASTA file and checks the corresponding
    embedding files in the provided directory. If the length of the embedding does not match the
    expected length, or if any errors occur during processing, the corresponding embedding file is removed.

    Usage:
    removeDeficientEmbedding("dataset/seq/Pruned_ecoli.fasta", "dataset/embd/ecoli", expectedEmbdLength)
    '''
    cnt = 0
    f = open(fastaAddress, 'r')
    while True:
        if cnt % 1000 == 0:
            print(cnt)
        cnt +=1
        pid = f.readline()
        pseq = f.readline()
        if pseq == "":
            break
        try:
            num_lines = 0
            for line in open('{}/{}.embd'.format(embdAddress,pid.strip()[1:])):
                num_lines += 1
                try:
                    embdLength = len(line.strip().split(':')[1].split(' '))
                except:
                    os.remove('{}/{}.embd'.format(embdAddress,pid.strip()[1:]))
                    break
                if embdLength != expectedEmbdLength:
                    #print("pid: {}, len seq: {}, num_lines: {}".format(pid.strip()[1:], len(pseq.strip()), embdLength))
                    os.remove('{}/{}.embd'.format(embdAddress,pid.strip()[1:]))
            if len(pseq.strip()) != num_lines:
                if os.path.exists('{}/{}.embd'.format(embdAddress,pid.strip()[1:])):
                    os.remove('{}/{}.embd'.format(embdAddress,pid.strip()[1:]))
                #print("pid: {}, len seq: {}, num_lines: {}".format(pid.strip()[1:], len(pseq.strip()), num_lines))
        except Exception as e:
            #print(e)
            pass
                
#removeDeficientEmbedding("dataset/seq/Pruned_human.fasta", "dataset/embd/bert/human", 1024)
#removeDeficientEmbedding("dataset/seq/interactomeV2.fasta", "dataset/embd/interactomeV2", 1024)

def checkMissed(dfAddress, embdAddress):
    """
    Check for missing protein embeddings in a specified directory.

    Parameters:
        dfAddress (str): The file path or address to the DataFrame containing a list of protein names.
        embdAddress (str): The directory path where protein embeddings are expected to be located.

    Returns:
        None

    Prints the names of proteins from the DataFrame at `dfAddress` that do not have corresponding embeddings in the
    `embdAddress` directory.

    Usage:
        checkMissed('dataset/pairs/fly_test.tsv', 'dataset/embd/fly')
    """
    protAB = protABgenerator(dfAddress)
    for protein in protAB:
        if not os.path.exists("{}/{}.embd".format(embdAddress, protein)):
            print(protein)

#checkMissed('dataset/pairs/fly_test.tsv', 'dataset/embd/fly')

def shuffler(dfAddress, saveAddress):
    '''
    shuffle dataset
    '''
    pairDf = pd.read_csv(dfAddress, sep='\t', 
                        names=['proteinA','proteinB','interact'])
    pairDf = pairDf.sample(frac=1).reset_index(drop=True)
    pairDf.to_csv(saveAddress,index=False, header=False, sep='\t')

#shuffler('dataset/pairs/human_validation_balanced.tsv', 'dataset/pairs/human_validation_balanced_shuffled.tsv')

def underSampler(dfAddress, saveAddress):
    '''
    Eliminate negative samples until they are equal to the positive ones. 
    '''
    df = pd.read_csv(dfAddress, sep='\t',
                     names=['proteinA','proteinB','interact'])
    balancedDF = df.groupby("interact").head(min(df["interact"].value_counts()))
    balancedDF.to_csv(saveAddress,index=False, header=False, sep='\t')

#underSampler('dataset/pairs/human_validation.tsv', 'dataset/pairs/human_validation_balanced.tsv')

def pairDatasetInfo(datasetAddress):
    """
    Counts the number of pairs in a dataset and prints the total count.

    Parameters:
    - datasetAddress (str): The file path or URL of the dataset containing pairs.

    This function reads the dataset and counts the number of pairs present. It then prints
    the total count of pairs.

    Example:
    >>> pairDatasetInfo("/path/to/pair_dataset.txt")
    5000

    Note: This function assumes that each line in the dataset represents a pair.

    """
    f = open(datasetAddress)
    pairCount = 0
    for line in f:
        pairCount+=1
    print(pairCount, end=' ')
#pairDatasetInfo('dataset/pairs/human_train.tsv')

def seqDatasetInfo(datasetAddress):
    """
    Analyzes a sequence dataset and prints information about protein sequences.

    Parameters:
    - datasetAddress (str): The file path or URL of the dataset containing protein sequences.

    This function reads the protein sequences from the specified dataset and calculates
    various statistics, including the total number of sequences, maximum length, minimum length,
    and average length. It then prints these statistics.

    Example:
    >>> seqDatasetInfo("/path/to/protein_dataset.fasta")
    1000 500 50 250.5

    Note: This function assumes that the dataset is in FASTA format, where each sequence is
    represented by a line starting with '>'. Lines not starting with '>' are considered as
    sequence lines.

    """
    f = open(datasetAddress)
    protCounter = 0
    protMaxLength = 0
    protMinLength = 1000
    protLengthAvg = 0
    for line in f:
        if ">" not in line:
            protCounter+=1
            protMaxLength = max(protMaxLength, len(line.strip()))
            protMinLength = min(protMinLength, len(line.strip()))
            protLengthAvg+=len(line.strip())
    protLengthAvg = protLengthAvg/protCounter

    print(protCounter,protMaxLength, protMinLength, protLengthAvg)
#seqDatasetInfo("dataset/seq/Pruned_human_train.fasta")
#seqDatasetInfo("dataset/seq/interactome.fasta")

def pair_creator(datasetAddress, outputAddress):
    """
    Creates pairs of protein identifiers (PIDs) from a FASTA file and saves them to an output file.

    Parameters:
    - datasetAddress (str): The file path to the input FASTA file containing protein sequences.
    - outputAddress (str): The file path where the pairs of protein identifiers will be stored.

    Returns:
    None

    This function reads protein sequences from the specified FASTA file and extracts the PIDs. 
    It then creates pairs of PIDs and writes them to the specified output file. Each line in the
    output file represents a pair of PIDs separated by a tab.

    Note: The function assumes that protein identifiers in the FASTA file are marked by '>' and
    are followed by the actual identifier. Make sure the input file follows this format.
    """
    f = open(datasetAddress)
    pid = []
    for line in f:
        if '>' in line:
            pid.append(line.strip()[1:])
    w = open(outputAddress, 'w')
    for index, pairA in enumerate(pid):
        for pairB in pid[index+1:]:
            w.write("{}\t{}\n".format(pairA, pairB))
#pair_creator("dataset/seq/interactomeV2.fasta", "dataset/pairs/interactomeV2.tsv")

def pair_dataset_divider(datasetAddress, outputAddress, partitionCapacity=83657):
    """
    Divide a pair dataset into partitions based on a specified capacity.

    Parameters:
    - datasetAddress (str): The file path to the input pair dataset.
    - outputAddress (str): The directory path where the partitioned files will be stored.
    - partitionCapacity (int): The maximum number of lines to include in each partition.

    Returns:
    None

    This function reads a pair dataset from the specified input file and divides it into partitions.
    Each partition contains a maximum of 'partitionCapacity' lines. The resulting partitions are
    saved as separate files in the specified output directory.

    The function creates output files with names in the format 'interactomeP1.tsv', 'interactomeP2.tsv', etc.
    """
    f = open(datasetAddress)
    partitionNumber = 1
    capacity = 1
    w=open("{}/interactomeP{}.tsv".format(outputAddress, partitionNumber), 'w')
    for line in f:
        w.write(line)
        capacity += 1
        if capacity > partitionCapacity:
            capacity = 1
            partitionNumber += 1
            w=open("{}/interactomeP{}.tsv".format(outputAddress, partitionNumber), 'w')

#pair_dataset_divider("dataset/pairs/interactomeV2.tsv", "dataset/pairs/interactomeV2", partitionCapacity=148160)  

def sequence_prunner(datasetAddress, outputAddress):
    f = open(datasetAddress)
    w = open(outputAddress, 'w')
    while True:
        label = f.readline()
        seq = f.readline().strip()
        if not label :
            break
        label = label.split('.')[0]
        w.write(label+'\n')
        w.write(seq[:800]+'\n')

#sequence_prunner("dataset/seq/humanProteomeLongestIsoforms.faa", "dataset/seq/InteractomeV2.fasta")

def compare_files(combinedAddress, predictAddress, labelAddress):
    """
    Compare the contents of three files and identify discrepancies.

    Parameters:
    - combined_address (str): Path to the combined file containing proteinA, proteinB, label, prediction.
    - predict_address (str): Path to the predict file containing proteinA, proteinB, prediction.
    - label_address (str): Path to the label file containing proteinA, proteinB, label.

    This function reads the contents of the three files and compares them. It prints lines from the predict_address
    file that are not present in the combined_address file and lines from the label_address file that are not present
    in the combined_address file.

    Returns:
    None
    """
    combinedPred = []
    combinedLabel = []
    combinedFile = open(combinedAddress)
    for line in combinedFile:
        if "proteinA" not in line:
            proteinA, proteinB, label, prediction, _ = line.strip().split('\t')
            combinedPred.append("{}\t{}\t{}".format(proteinA, proteinB, round(float(prediction), 4)))
            combinedLabel.append("{}\t{}\t{}".format(proteinA, proteinB, label))
    predictFile = open(predictAddress)
    for line in predictFile:
        proteinA, proteinB, prediction = line.strip().split('\t')
        line = "{}\t{}\t{}".format(proteinA, proteinB, round(float(prediction), 4))
        if line not in combinedPred:
            print(line)
    labelFile = open(labelAddress)
    for line in labelFile:
        line = line.strip()
        if line not in combinedLabel:
            print(line)

#compare_files("tt/ecoli_labeled.tsv", "tt/ecoli.tsv", "dummy/ecoli_test.tsv")

def filter_incomplete_results(labelDirectory, predictedDitectory):
    """
    Check the integrity of partitions in a directory.

    This function iterates through partition numbers ranging from a range
    and checks whether corresponding files exist in the provided directories.
    For each partition, it compares the number of lines in a label file and
    a predicted file to ensure they match, except for a possible header line.
    
    Args:
        predictedDirectory (str): The directory where predicted files are located.
        labelDirectory (str): The directory where label files are located.
        
    Prints:
        If the number of lines in the label file is not equal to one less than
        the number of lines in the predicted file, it prints the paths of the
        predicted file, the number of lines in the label file, and the number
        of lines in the predicted file.

    Returns:
        None

    Example:
    >>> filter_incomplete_results("path/to/labels", "path/to/predicted")
    """
    for partitionNumber in range(0,2500):
        partition = "interactomeP"+str(partitionNumber)
        if os.path.isfile('{}/{}.tsv'.format(predictedDitectory, partition)):
            labelLength = os.popen("wc -l {}.tsv".format(labelDirectory+partition)).read().split()[0]
            predictedLength = os.popen("wc -l {}/{}.tsv".format(predictedDitectory, partition)).read().split()[0]
            #break
            if int(labelLength) != int(predictedLength)-1:
                print("{}/{}.tsv".format(predictedDitectory, partition), labelLength, predictedLength)
                        
#filter_incomplete_results("dataset/pairs/interactomeV2/", "results/interactomeV2/")
def drop_label(inputCSV):
    df = pd.read_csv(inputCSV, delimiter='\t')
    df.drop(columns=['predictionLabel'], inplace=True)
    df.to_csv('results/completeInteractome.csv', index=False, sep='\t')


#drop_label('results/interactomeV2Complete.csv')

def drop_label_concate(inputDir, outputAddress):
    w = open(outputAddress,'w')
    w.write("proteinA\tproteinB\tprediction\n")
    pos = 0
    for fileAddress in glob.glob(inputDir+"/*"):
        f = open(fileAddress)
        f.readline()
        for line in f:
            line=line.strip().split('\t')#[:-1]
            pos += int(line[-1])
            line = line[:-1]
            w.write('\t'.join(line)+'\n')
    print(pos)

#drop_label_concate('results/interactomeV2', "results/completeInteractome.csv")


def oneLiner(inputFastaAddress, outputFastaAddress):
    f = open(inputFastaAddress, 'r')
    w = open(outputFastaAddress, 'w')
    for line in f:
        line = line.strip()
        if '>' == line[0]:
            w.write('\n'+line+'\n')
        else:
            w.write(line)

#oneLiner('dataset/seq/Plasminogens_dedup.fasta', 'dataset/seq/Plasminogens.fasta')
#sequence_prunner('dataset/seq/Plasminogens.fasta', 'dataset/seq/Pruned_Plasminogens.fasta')
#removeDeficientEmbedding("dataset/seq/Pruned_mouse.fasta", "dataset/embd/ankh/mouse", 1536)
#seqDatasetInfo('dataset/seq/prunedinteractomeSpecies.fasta')
#pair_creator("dataset/seq/Pruned_Plasminogens.fasta", "dataset/pairs/Plasminogens.tsv")
#pair_dataset_divider("dataset/pairs/interactomeSpecies.tsv", "dataset/pairs/interactomeSpecies", partitionCapacity=73088)  
#removeDeficientEmbedding("dataset/seq/Pruned_mouse.fasta", "dataset/embd/albert/mouse/", 4096)
#filter_incomplete_results("dataset/pairs/interactomeSpecies/", "results/interactomeSpecies/")
#drop_label_concate('results/Plasminogens/', "results/completePlasminogens.csv")
