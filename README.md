## Molecular Set Transformer
A deep learning model, namely Molecular Set Transformer, was designed for enabling high-throughput co-crystal screening for any type of molecular pairs. The model is able to provide score and uncertainty for any given molecular pair based on its probability to form a multicomponent cystal.
<img src="https://github.com/katerinavr/cocrystals/blob/master/figures/TOC.png" width="800" height="400">

## Extract co-crystals from the Cambridge Structural Database (CSD)
You need the CSD licence (https://www.ccdc.cam.ac.uk/) to use this script. You can extract multicomponent crystals after screening the whole > 1 million CSD database. If you do not want to include the solvents , you can use the option  `--no_solvents`

    python extract_cocrystals/extract_cocrystals.py --no_solvents

Our extracted csd co-crystals dataset using CSD 2020 can be found in `csd_data/csd_cocrystals2020.csv`
The training co-crystals dataset was created after removing the duplicate pairs, i.e., pairs found both in the csd_cocrystals2020 and in the benchmark data and is referred as `csd_data/csd_training_set.csv`

    python extract_cocrystals/drop_duplicates.py

## Benchmarks (Publically available co-crystal screening data)
All the is-silico co-crystal screening datasets gathered from literature can be found on the validation_data folder and are described below in chronological order:

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

## Using pretrained models to rank any test data

    python eval.py --model <model> --dataset_name <dataset_name> --save_dir <save_dir> --get_plots --threshold <threshold>

- `model` The molecular embeddings to use. Choose among:
    - gnn
    - morgan
    - chemberta
    - mordred
- `dataset_name` The location of the .csv file including the molecular pairs 
- `save_dir` The folder to save the generated files 
- `threshold` Only needed if plotting the results and have some known labels

Example

    python eval.py --model gnn --dataset_name meps --save_dir results_folder --get_plots --threshold 0.85

## Or train your own models based on the prefered molecular representation

    python train.py --model <model> --training_data <data> --save_dir <save_dir> -n_epochs <epochs> -lr <learning_rate>

- `model` The molecular embeddings to use. Choose among:
    - gnn
    - morgan
    - chemberta
    - mordred
- `training_data` The location of the .csv file containing the molecular pairs for training
- `n_epochs` The number of epochs to use for training    
- `lr` Learning rate

## Interpretability with Shapley
A notebook that explains how you can use SHAP to analyse and interpret the predictions of our machine learning models is provided here: `notebooks/interpretability.ipynb` 

## Alternatively use Streamlit to get quick predictions for any given SMILES pair 

just click on the link: [https://share.streamlit.io/katerinavr/streamlit/app.py](https://share.streamlit.io/katerinavr/streamlit/app.py)
