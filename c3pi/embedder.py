from bio_embeddings.embed import ProtTransT5XLU50Embedder 
import os
import random

def process_sequences(fastaAddress, outputAddress):
    fin = open(fastaAddress, "r")
    embedder = ProtTransT5XLU50Embedder()    
    while True:
        line_PID = fin.readline().strip()[1:]  # Skip '>' or similar
        line_Pseq = fin.readline().strip()[:900]

        if not line_Pseq:
            break

        output_file = os.path.join(outputAddress, f"{line_PID}.embd")
        
        try:
            with open(output_file, 'x') as w:
                pass  
        except FileExistsError:
            continue

        # Process and write embeddings
        with open(output_file, 'a') as w:
            embedding = embedder.embed(line_Pseq)
            for cnt, aa in enumerate(line_Pseq):
                w.write(f"{aa}:{' '.join(str(x) for x in embedding[cnt])}\n")
process_sequences("dataset/seq/noch.fasta","dataset/embd/noch/")