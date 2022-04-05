## Molecular Set Transformer
A deep learning model, namely Molecular Set Transformer, was designed for enabling high-throughput co-crystal screening for any type of molecular pairs. The model is able to provide score and uncertainty for any given molecular pair based on its probability to form a multicomponent cystal.

## Extract co-crystals from the Cambridge Structural Database (CSD)
You need the CSD licence (https://www.ccdc.cam.ac.uk/) to use this script. You can extract multicomponent crystals after screening the whole > 1 million CSD database. If you do not want to include the solvents , you can use the option  `--no_solvents`

    python extract_cocrystals/extract_cocrystals.py --no_solvents

Our extracted csd co-crystals dataset using CSD 2020 can be found in `csd_data/csd_cocrystals2020.csv`

## Benchmarks (Publically available co-crystal screening data)
All the is-silico co-crystal screening datasets gathered from literature can be found on the validation_data folder and are described below:

|               |     Dataset reference                                                         |     Number or data                           |     Other methods tested on these   datasets    |
|---------------|-------------------------------------------------------------------------------|----------------------------------------------|-------------------------------------------------|
|     1         |     Wicker et al, CrystEngComm,   2017, 19, 5336–5340                         |     680 (408   negatives + 272 positives)    |                                                 |
|     2         |     Grecu   et al, Cryst.   Growth Des., 2014, 14, 165–171                    |     432 (300   negatives + 132 positives)    |     MEPS, COSMO-RS                              |
|     3         |     Wood et al, CrystEngComm,   2014, 16, 5839–5848                           |     38 (36   negatives + 2 positives)        |                                                 |
|     4         |     Vriza et al, Chem. Sci., 2021, 12, 1702–1719                              |     6 (4 negatives + 2 positives)            |                                                 |
|     5         |     Mapp et al, Cryst.   Growth Des., 2017, 17, 163–174     Khalaji et al     |     108 (90 negatives + 18 positives)        |     MC                                          |
|     6         |     Przybyłek   et al, Cryst.   Growth Des., 2019, 19, 3876–3887              |     712 (104 negatives + 608 positives)      |                                                 |
|     7         |     Przybyłek   and P. Cysewski, Cryst.   Growth Des., 2018, 18, 3524–3534    |     226 (58 negatives + 168 positives)       |                                                 |
|     8         |     Aakeröy   et al, Cryst.   Growth Des., 2009, 9, 432–441                   |     82 (17 negatives + 65 positives)         |                                                 |
|     9         |     Devogelaer   et al, Cryst.   Growth Des., 2021, 21, 3428–3437             |     30 (18 negatives + 12 positives)         |                                                 |
|     10        |     Wu et al, Cryst.   Growth Des., 2021, 21, 4531–4546                       |     63 (22 negatives + 41 positives)         |     COSMO-RS, MC                                |


## Using pretrained models to rank any test data

    #Molecular descriptors model:

    python mordred_descriptors.py -dataset_name data/grecu_example.csv -save_dir grecu_data -threshold 0.81

    #Fingerprint model:

    python src/fingerprint_representation.py -dataset_name data/grecu_example.csv -save_dir grecu_data -threshold 0.84

    #GNN model:

    python src/gnn_fingerprint.py -dataset_name data/grecu_example.csv -save_dir grecu_data -threshold 0.84
    
    # BERT model:

    python src/bert_fingerprint.py -dataset_name data/grecu_example.csv -save_dir grecu_data -threshold 0.84
    
 
## Or train your own models based on the prefered molecular representation

Molecular descriptors model:

    python train_mordred_descriptors.py -dataset_name [dataset_name] -model_save_dir [directory to save your model]

Fingerprint model:

    python train_moorgan_descriptors.py -dataset_name [dataset_name] -model_save_dir [directory to save your model]

GNN model:

    python train_gnn_descriptors.py -dataset_name [dataset_name] -model_save_dir [directory to save your model]
    
BERT model:

    python train_bert_descriptors.py -dataset_name [dataset_name] -model_save_dir [directory to save your model]


## You can use Streamlit to get quick predictions for any given SMILES pair 

    just click on the link: https://share.streamlit.io/katerinavr/streamlit/app.py