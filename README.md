# C3PI

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

C3PI is a protein-protein interaction prediction model that combines Contextual embbeding, 
Jigsaw puzzles and ensumble architecture together.


## Getting Started

To run the code, follow these steps:

1. Clone the repository to your Linux machine:

    ```bash
    git clone https://github.com/lucian-ilie/C3PI.git
    cd C3PI
    ```

2. Create virtual environments for Python 3.10 (ppiENV) and Python 3.8 (embdENV):

    ```bash
    python3.10 -m venv mainENV
    python3.8 -m venv embedENV
    ```

    Activate the virtual environments:

    ```bash
    source ppiENV/bin/activate
    source embdENV/bin/activate
    ```

3. Install dependencies for embedENV:

    ```bash
    pip install -r requirements_embed.txt
    ```

    Run the script for computing embeddings:

    ```bash
    python embedCreatorT5Functional.py
    ```

4. Install dependencies for mailENV:

    ```bash
    pip install -r requirements_main.txt
    ```

    Train the model using either:

    ```bash
    python trainCNN1D.py
    ```

    or

    ```bash
    python trainCNN2D.py
    ```

5. After training, use the prediction scripts for each species:

    ```bash
    python predict.py
    ```

6. For users familiar with High Performance Computing (HPC) or Advanced Research Computing (ARC), you can use the provided batch scripts:

    - Use `grtrainv2.sh` to train.
    - Use `grtest.sh` to predict.
    - Use `grEmbeddCreator.sh` to compute embeddings.

    Make sure to edit the files and input your appropriate environment and username.

7. The `untils.py` file contains functions used during development, each with a descriptive docstring.

8. `evaluation.py` computes metrics for model predictions.

9. `curvePlot.py` used to plot ROC an PR curves.
  

Feel free to explore, and if you encounter any issues, refer to the documentation or reach out for assistance.
   

## Project Structure
```bash
C3PI/
│
├── dataset/
│   ├── embd/
│   │   ├── ecoli/  
│   │   ├── fly/  
│   │   ├── human/  
│   │   ├── interactome/  
│   │   ├── mouse/  
│   │   ├── plasminogens/  
│   │   ├── S_venezuelae/  
│   │   ├── worm/  
│   │   └── yeast/
│   ├── pairs/
│   └── seq/
│  
│
├── models/
│   ├── README.md
│   ├── CNN1D.weight
│   └── CNN2D.weight
│
└── results/
    ├── ecoli_C3PI.tsv  
    ├── fly_C3PI.tsv  
    ├── human_C3PI.tsv  
    ├── mouse_C3PI.tsv  
    ├── worm_C3PI.tsv  
    └── yeast_C3PI.tsv

```
## Dependencies

### Python Dependencies

To install all required Python libraries, you'll need to activate the respective virtual environments and install the libraries listed in the `requirements_embed.txt` and `requirements_main.txt` files. These dependencies ensure that the code executes properly.

Here's a brief overview of some key dependencies:

- `TensorFlow` and `Keras` - for building and training the neural network models.
- `scikit-learn` - for various machine learning tools.
- `pandas` - for data manipulation and analysis.
- `numpy` - for numerical operations.
- `matplotlib` and `seaborn` - for plotting graphs and visualizing data.

Make sure your system meets these requirements to ensure seamless setup and execution of the model.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

