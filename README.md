# baRNAcle - Leveraging 2.5D Graph Representations and Relational Graph Convolutional Networks for RNA-Ligand Binding Site Prediction

## Introduction


## Requirements
x3dna-dssr executable in path  
ViennaRNA (version 2.4.18 via conda installation)  
varnaapi (version 0.1.0)  
rnalgib dependencies (do ```pip install rnaglib``` and then ```pip uninstall rnaglib``` to install the dependencies without using a deprecated rnaglib version)  
BayesPairing2 dependencies (installed following instructions at https://jwgitlab.cs.mcgill.ca/sarrazin/rnabayespairing2/tree/master and then uninstalled)  

## Data Preparation
Data preparation is a multistep process described in the baRNAcle report. Since data preparation is a very computationally expensive step, we offer two options.
1. A pre-annotated dataset of 2.5D graphs is included in the ```data.tgz``` file which needs to be un-tar-ed before use. Namely, the cleaned_bp2_annotated_15cutoff directory is ready for model training with no further processing needed.
2. Any set of mmcif files can be converted into annotated 2.5D graphs using the ```run_data_generation_pipeline``` method in ```generate_graphs.py```. To run this method, all requirements need to be pre-installed. The pipeline can be run from the command line from the repo's home directory as follows:    
    * ```$ python3 generate_graphs.py <path_to_mmcif_files> <binding_pocket_cutoff_distance>```    
    * where ```<path_to_mmcif_files>``` is a path in the form of a string leading to the directory of mmcif files to transform to 2.5D graphs,  
    * and ```<binding_pocket_cutoff_distance>``` is an integer corresponding to the cutoff distance defining which nodes are part of binding pockets(default = 15).  
  
    * upon termination of this command, two output repositories of 2.5D graphs should exist in the ```temp_dir``` directory. ```RNAchain_graphs``` and ```monomer_graphs``` should contain 2.5D graphs representing RNA chains from a mix of RNA multimers and monomers or only from RNA monomers, respectively.
## Model Training and Modification

We offer the possibility of building different models. Namely, there are 4 main hyperparameters to play with: the number of layers, the type of output layer, the number of epochs, and the learning rate. A detailed analysis of the hyperparameter tuning performed to obtain the best performing model is described in the baRNAcle report, where we obtain best results for a model with 4 layers, no convolutional output, 0.01 learning rate and 50 training epochs.  
To run model training, the script can be run from command line from the repo's home directory as follows:
* ```$ python3 simple_gnns.py <graph_path> <output_layer> <num_layers> <epochs> <data_split> <save>``` where 
* <graph_path> is the path to the 2.5D graphs directory
* <output_layer> is a boolean, True means the ouput layer is convolutional, False means it's linear
* <num_layers> is an int describing the number of RGCN (and linear if output_layer = False) stacked on top of one another
* <epochs> is the number of training epochs
* <data_split> defines the type of dataset the model is being tested on. For data_split=0, the model is trained and tested on a mixed dataset of RNA chains coming from monomers and multimers. For data_split=1, the model is trained on monomers and multimers, but is tested only on monomers. Finally, for data_split=2, the model is trained on monomers and multimers, but is tested only on multimers. If data_split=1 or 2, then an additonal argument should be passed to the internal method ```run_gnn```. The additional parameter, indep_graphs_path, is a path to the set of monomer chains.
* <save> is a boolean that determines whether the model weights are saved or not. The model is saved to "./models/gnn_weights.pt"  
  

The results of training are written to a csv file stored in "./output/results.csv"  
To get a **quick start**, one can directly train a model by running the following commands:  
```$ tar zxvf data.tgz```    
```$ python3 simple_gnns.py './data/cleaned_bp2_annotated_15cutoff' 4 50 0 False```  
