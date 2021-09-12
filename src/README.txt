-The dataset should be given in same format as the data/grecu_example.csv
Name1, Smiles1, Name2, Smiles2 on the main()

-  For doing the predictions based on the fingerprints:
   e.g python fingerprint_representation.py -dataset_name data/grecu_example.csv -save_dir grecu_data -threshold 0.8

-  For doing the predictions based on Mordred:
   e.g python mordred_descriptors.py -dataset_name data/grecu_example.csv -save_dir grecu_data -threshold 0.8

The main arguments you need are: 
dataset_name --> the dataset name you want to test on and is located in the data folder
save_dir --> the directory you want to save your results (a csv file with the scores for each pair, a color-rank plot
and the confusion matrix)
threshold (optional, default=0.8): the threshold you want to set for getting the confucion matrix 
