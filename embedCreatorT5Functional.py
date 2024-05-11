from bio_embeddings.embed import ProtTransT5XLU50Embedder 
import numpy as np
import time
import random
import os


def embeddingCalculator(fastaAddress, outputAddress):
    """
    Calculates and stores protein embeddings from a FASTA file.

    Parameters:
    - fastaAddress (str): The file path to the input FASTA file containing protein sequences.
    - outputAddress (str): The directory path where the embedding files will be stored.

    Returns:
    None

    This function reads protein sequences from the specified FASTA file, calculates embeddings
    using ProtTransT5XLU50Embedder, and stores the results in separate files for each protein in
    the specified output directory. If an embedding file already exists, it skips the calculation
    for that protein. A brief random sleep is introduced to avoid duplications.

    Note: Make sure to have ProtTransT5XLU50Embedder imported and installed before using this function.
    """
    fin = open(fastaAddress, "r")
    embedder = ProtTransT5XLU50Embedder()    
    while True:
        line_PID = fin.readline().strip()[1:]
        line_Pseq = fin.readline().strip()
        if not line_Pseq:
            break
        if os.path.isfile("{}/{}.embd".format(outputAddress, line_PID)):
            continue
        time.sleep(round(random.uniform(0,3),3))
        if not os.path.isfile("{}/{}.embd".format(outputAddress, line_PID)):
            w = open("{}/{}.embd".format(outputAddress, line_PID), 'w')
            w.close()
            w = open("{}{}.embd".format(outputAddress, line_PID), 'a')
            embedding = embedder.embed(line_Pseq)
            for cnt, aa in enumerate(line_Pseq):
                w.write(aa+':')
                w.write(' '.join([str(x) for x in embedding[cnt]]))
                w.write('\n')

embeddingCalculator("dataset/seq/Pruned_Plasminogens.fasta","dataset/embd/Plasminogens/")