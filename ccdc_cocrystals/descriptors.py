import streamlit as st
import pandas as pd
import numpy as np
import base64
from rdkit import Chem
from mordred import Calculator, descriptors
from one_class import *
import itertools

def mordred_descriptors():
    sentence = st.text_area('Input your SMILES here:',  key="1") 
    lista=[]
    lista.append(sentence.split())

    calc = Calculator(descriptors, ignore_3D=True)
    if len(sentence.split()) > 0:
        mol = [Chem.MolFromSmiles(x) for x in sentence.split()]
        descriptors_mol1 =[]
        for mol in sentence.split():
    	    try:
                descriptors_mol1.append(calc(Chem.MolFromSmiles(mol)))
    	    except TypeError:
                descriptors_mol1.append('none')
        dataset1 = pd.DataFrame(descriptors_mol1)
        df1 = pd.DataFrame(dataset1.values, columns=calc.descriptors)#.to_csv('dataset1.csv', index=False)
        df = pd.concat([pd.DataFrame(sentence.split(), columns=['smiles']), df1], axis=1)
        df.to_csv('data/df_user.csv', index=False)
        st.write(df)
        if st.button('Download Dataframe as CSV',key="2"):
            tmp_download_link = download_link(df, 'descriptors.csv', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True, key="3")
    

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def get_score():
    dataset1 = pd.read_csv('data/dataset1.csv')
    dataset2 = pd.read_csv('data/dataset2.csv')
    df1=dataset1.iloc[:,2:]
    df1 = df1.fillna(0)
    df2=dataset2.iloc[:,2:]
    df2 = df2.fillna(0)
    df1 = df1.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    df2 = df2.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)  
    X_scaler = MinMaxScaler()
    df_concat = pd.concat([df1, df2])
    df_concat = df_concat.drop_duplicates(keep='first')
    numerical_cols = df_concat.columns[:]
    df_scaled = pd.DataFrame(X_scaler.fit(df_concat), columns=numerical_cols, index=df_concat.index)
    numerical_cols = df2.columns[:]
    df1_scaled =  pd.DataFrame(X_scaler.transform(df1[numerical_cols]), columns=numerical_cols, index=df1.index)
    df2_scaled = pd.DataFrame(X_scaler.transform(df2[numerical_cols]), columns=numerical_cols, index=df2.index)
    # Final bidirectional concatenated dataset, after feature selection and scaling 
    df = concat_bidirectional(df1_scaled,df2_scaled)
    labelled = pd.concat([df1_scaled, df2_scaled], axis=1)

    # Import the unlabeled dataset
    #unlabeled = pd.read_csv('data/zinc15_dataset.csv')
    df_user=pd.read_csv('data/df_user.csv', index_col=False)
    unlabeled=df_user
    unlabeled=unlabeled.iloc[:,:]
    val = df_user['smiles'].values
    length = len(val)
    pairs = [[val[i],val[j]] for i in range(length) for j in range(length) if i!=j ]
    # Remove the duplicate structures
    no_dups = []
    for pair in pairs:
        if not any(all(i in p for i in pair) for p in no_dups):
            no_dups.append(pair)
    pairs = pd.DataFrame(no_dups)
    keys = unlabeled['smiles'].values
    values = unlabeled.iloc[:, 1:].values
    d = {key:value for key, value in zip(keys, values)}
    mol1_data= list()
    for mol1 in pairs[0]:       
        mol1_data.append(d[mol1])    
    mol1_data = pd.DataFrame(mol1_data, columns = unlabeled.iloc[:, 1:].columns.values)   
    mol2_data= list()
    for mol2 in pairs[1]:   
        mol2_data.append(d[mol2])
    mol2_data = pd.DataFrame(mol2_data, columns= unlabeled.iloc[:, 1:].columns.values) 
    final_1 = pd.concat([pairs[0],mol1_data],axis=1)
    final_1 = final_1.fillna(0)
    final_2 = pd.concat([pairs[1],mol2_data],axis=1)
    final_2 = final_2.fillna(0)
    unlab=pd.concat([pairs[0], pairs[1]], axis=1)
    final_1 = final_1.replace({'#NUM!': 0})
    final_2 = final_2.replace({'#NUM!': 0})
    final_11 = final_1.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    final_22 = final_2.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    # Standarize the unlabeled data based on the labelled
    final_1_scaled = pd.DataFrame(X_scaler.transform(final_11.iloc[:,1:]))
    final_2_scaled = pd.DataFrame(X_scaler.transform(final_22.iloc[:,1:]))
    uf=pd.concat([final_1_scaled, final_2_scaled], axis =1)

    torch.manual_seed(0)
    deepSVDD.build_network = build_network
    deepSVDD.build_autoencoder = build_autoencoder
    deep_SVDD = deepSVDD.DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    net_name='model_checkpoint_final_cpu.pth'
    deep_SVDD.set_network(net_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deep_SVDD.load_model('deep_one_class/model_checkpoint_final_mypc.pth')
    y_scaler1 = MinMaxScaler()

    lab = score(deep_SVDD, df.iloc[:,1:].values)*-1  #.cpu().detach().numpy()*-1 
    lab= y_scaler1.fit_transform(lab.reshape(-1,1))
    lab=pd.DataFrame(lab, columns=['train_score'])
    unlab_ = score(deep_SVDD, uf.iloc[:,:].values).cpu().detach().numpy()*-1
    unlab_ = y_scaler1.transform(unlab_.reshape(-1,1))
    unlab_final=pd.DataFrame(unlab_, columns=['your_score'] )
    unlab_final=pd.concat([pd.DataFrame(unlab.values, columns=['mol1', 'mol2']), pd.DataFrame(unlab_, columns=['test_score'] )], axis=1)
    print(lab.describe())
    print(unlab_final.describe())
    
    st.write(unlab_final)



def get_score_our_data():
    dataset1 = pd.read_csv('data/dataset1.csv')
    dataset2 = pd.read_csv('data/dataset2.csv')
    df1=dataset1.iloc[:,2:]
    df1 = df1.fillna(0)
    df2=dataset2.iloc[:,2:]
    df2 = df2.fillna(0)
    df1 = df1.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    df2 = df2.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)  
    X_scaler = MinMaxScaler()
    df_concat = pd.concat([df1, df2])
    df_concat = df_concat.drop_duplicates(keep='first')
    numerical_cols = df_concat.columns[:]
    df_scaled = pd.DataFrame(X_scaler.fit(df_concat), columns=numerical_cols, index=df_concat.index)
    numerical_cols = df2.columns[:]
    df1_scaled =  pd.DataFrame(X_scaler.transform(df1[numerical_cols]), columns=numerical_cols, index=df1.index)
    df2_scaled = pd.DataFrame(X_scaler.transform(df2[numerical_cols]), columns=numerical_cols, index=df2.index)
    # Final bidirectional concatenated dataset, after feature selection and scaling 
    df = concat_bidirectional(df1_scaled,df2_scaled)
    labelled = pd.concat([df1_scaled, df2_scaled], axis=1)
    unlabeled = pd.read_csv('data/zinc15_dataset.csv')
    df_user = pd.read_csv('data/df_user.csv')
    total_df= pd.concat([unlabeled.iloc[:,1:], df_user])

    unlab_smiles = unlabeled.smiles.values[:5]  #select unlabelled 
    user_smiles = df_user.smiles.values
    list1=unlab_smiles
    list2=user_smiles
    combination = [(i, j) for i in list1 for j in list2]
    combi = pd.DataFrame(combination, columns=['mol1', 'mol2'])
    #st.write(total_df)
    keys = total_df['smiles'].values
    values =total_df.iloc[:, 1:].values
    d = {key:value for key, value in zip(keys, values)}

    mol1_data= list()
    for mol1 in combi.mol1.values:       
        mol1_data.append(d[mol1])    
    mol1_data = pd.DataFrame(mol1_data, columns = total_df.iloc[:, 1:].columns.values)   
    mol2_data= list()
    for mol2 in combi.mol2.values:   
        mol2_data.append(d[mol2])
    mol2_data = pd.DataFrame(mol2_data, columns= total_df.iloc[:, 1:].columns.values) 
    final_1 = pd.concat([combi.mol1,mol1_data],axis=1)
    final_1 = final_1.fillna(0)
    final_2 = pd.concat([combi.mol2,mol2_data],axis=1)
    final_2 = final_2.fillna(0)
    unlab=pd.concat([combi.mol1, combi.mol2], axis=1)
    final_1 = final_1.replace({'#NUM!': 0})
    final_2 = final_2.replace({'#NUM!': 0})
    final_11 = final_1.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    final_22 = final_2.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    # Standarize the unlabeled data based on the labelled
    final_1_scaled = pd.DataFrame(X_scaler.transform(final_11.iloc[:,1:]))
    final_2_scaled = pd.DataFrame(X_scaler.transform(final_22.iloc[:,1:]))
    uf=pd.concat([final_1_scaled, final_2_scaled], axis =1)

    torch.manual_seed(0)
    deepSVDD.build_network = build_network
    deepSVDD.build_autoencoder = build_autoencoder
    deep_SVDD = deepSVDD.DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    net_name='model_checkpoint_final_cpu.pth'
    deep_SVDD.set_network(net_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deep_SVDD.load_model('deep_one_class/model_checkpoint_final_mypc.pth')
    y_scaler1 = MinMaxScaler()

    lab = score(deep_SVDD, df.iloc[:,1:].values)*-1  #.cpu().detach().numpy()*-1 
    lab= y_scaler1.fit_transform(lab.reshape(-1,1))
    lab=pd.DataFrame(lab, columns=['train_score'])
    unlab_ = score(deep_SVDD, uf.iloc[:,:].values).cpu().detach().numpy()*-1
    unlab_ = y_scaler1.transform(unlab_.reshape(-1,1))
    unlab_final=pd.DataFrame(unlab_, columns=['your_score'] )
    unlab_final=pd.concat([pd.DataFrame(unlab.values, columns=['mol1', 'mol2']), pd.DataFrame(unlab_, columns=['test_score'] )], axis=1)
    print(lab.describe())
    print(unlab_final.describe())
    
    st.write(unlab_final)
