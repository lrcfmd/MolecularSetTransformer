# Molecular Set Transformer
A deep learning model, namely Molecular Set Transformer, was designed for enabling high-throughput co-crystal screening for any type of molecular pairs. The model is able to provide score and uncertainty for any given molecular pair based on its probability to form a multicomponent crystal.
<img src="https://github.com/katerinavr/cocrystals/blob/master/figures/TOC.png" width="800" height="400">

# Installation
We recommend installing the package by following the instructions below.
```
conda create --name cocrystals python=3.8.8
conda activate cocrystals
git clone https://github.com/lrcfmd/MolecularSetTransformer.git
cd MolecularSetTransformer
pip install -r requirements.txt
```
either cloning the environment from the cocrystals.yaml file
```
conda create --name myclone --clone cocrystalsy
```

# Extract co-crystals from the Cambridge Structural Database (CSD)
You need the CSD licence (https://www.ccdc.cam.ac.uk/) to use this script. You can extract multicomponent crystals after screening the whole > 1 million CSD database. If you do not want to include the solvents , you can use the option  `--no_solvents`

    python extract_cocrystals/extract_cocrystals.py --no_solvents

Our extracted csd co-crystals dataset using CSD 2020 can be found in `csd_data/csd_cocrystals2020.csv`
The training co-crystals dataset was created after removing the duplicate pairs, i.e., pairs found both in the csd_cocrystals2020 and in the benchmark data and is referred as `csd_data/csd_training_set.csv`

    python extract_cocrystals/drop_duplicates.py

The code for generating the co-crystals tree map can be found in the ```TMAP-cocrystals``` folder. The PPD code for calculating the distances between the co-crystals can be found in https://github.com/dwiddo/average-minimum-distance

# Benchmarks (Publically available co-crystal screening data)
All the is-silico co-crystal screening datasets gathered from literature can be found on the validation_database folder and are described below in chronological order:

|               |     Dataset reference                                                         |     Number or data                           |     computational methods tested on these   datasets    |
|---------------|-------------------------------------------------------------------------------|----------------------------------------------|-------------------------------------------------|
|     1         |     Karki et al, CrystEngComm, 2010, 12, 4038–4041                            |     75 (73 negatives + 2 positives)          |                                                 |
|     2         |     Grecu   et al, Cryst. Growth Des., 2014, 14, 165–171                      |     432 (300 negatives + 132 positives)      |   MEPS, COSMO-RS                                 |
|     3         |     Wicker et al, CrystEngComm,   2017, 19, 5336–5340                         |     680 (408   negatives + 272 positives     |                                                 |
|     4         |     Mapp et al, Cryst. Growth Des., 2017, 17, 163–174                         |     108 (90 negatives + 18 positives)        |   MC                                             |
|     5         |     Przybyłek et al, Cryst. Growth Des., 2018, 18, 3524–3534                  |     226 (58 negatives + 168 positives)       |                                                 |
|     6         |     Przybyłek et al, Cryst.   Growth Des., 2019, 19, 3876–3887                |     712 (104 negatives + 608 positives)      |                                                 |
|     7         |     L. Roca-Paixão et al, CrystEngComm, 2019, 21, 6991–7001                   |     21 (8 negative and 13 positives)         |                                                 |
|     8         |     Aakeröy   et al, Cryst.   Growth Des., 2019, 9, 432–441                   |     82 (17 negatives + 65 positives)         |                                                 |
|     9         |     M. Khalaji et al, Cryst. Growth Des., 2021, 21, 2301–2314                 |     19 (9 negatives and 10 positives)        |                                                 |
|     10        |     Vriza et al, Chem. Sci., 2021, 12, 1702–1719                              |     6 (4 negatives + 2 positives)            |                                                 |
|     11        |     Devogelaer et al, Cryst. Growth Des., 2021, 21, 3428–3437                 |     30 (18 negatives + 12 positives)         |                                                 |
|     12        |     Wu et al, Cryst. Growth Des., 2021, 21, 4531–4546                         |     63 (22 negatives + 41 positives)         |   COSMO-RS, MC                                   |
|     13        |     J. Yuan et al, CrystEngComm, 2021, 23, 6039–6044                          |     16 (9 negatives + 7 positives)           |   COSMO-RS                                       |


# Train your own models based on the prefered molecular representation

    python train.py --model <model> --training_data <data> --save_dir <save_dir> -n_epochs <epochs> -lr <learning_rate>

- `model` The molecular embeddings to use. Choose among:
    - gnn
    - morgan
    - chemberta
    - mordred
- `training_data` The location of the .csv file containing the molecular pairs for training
- `save_dir` The directory to save the trained model 
- `n_epochs` The number of epochs to use for training    
- `lr` Learning rate
- `use wandb` We use Weight&Biases to monitor the hyperparameters during the training. This not a strict requirement but it is suggested. If you want to use it, you need to set up a `config.json` file with the hyperparameters and to create an account [here](https://wandb.ai/) and install the Python package:

    ```  
    pip install wandb
    ```

Example

    python train.py --model gnn --training_data csd_data/csd_cocrystals2020.csv --save_dir pretrained_models -n_epochs 100 -lr 0.001

# Notebooks
- ```cocrystal_statistics.ipynb``` Contains the code to reproduce the plots regarding the statistical analysis of the current co-crystal related research
- ```GNN_train_evaluate_example.ipynb``` Contains the code to train/evaluate and reproduce the plots of the paper for the GNN model.


# Comparing with other co-crystal screening methods 
- ```ccgnet_reports``` Contains the outcome from the tests using the CCGnet model(github repo)
-MEPS and COSMO-RS data 

# Use our pretrained models to get quick predictions for any given SMILES pair 

Click on the link: [Co-crystal screeening GUI](https://share.streamlit.io/katerinavr/streamlit/app.py)

The full code and dependencies can be found to the following repository: https://github.com/katerinavr/streamlit