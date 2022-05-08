import os
import torch
import sys
import numpy as np
import random as rand

from lib.rnaglib_main.rnaglib.learning import models, learn
from lib.rnaglib_main.rnaglib.data_loading.graphloader import GraphDataset, GraphLoader
from lib.rnaglib_main.rnaglib.learning.learning_utils import LearningRoutine, send_graph_to_device
from lib.rnaglib_main.rnaglib.utils import misc
from sklearn.metrics import roc_auc_score
from csv import writer

def run_gnn(json_graphs_path,
            convolutional = True,
            layers = 1,
            lr = 0.001,
            epochs = 25,
            csv_path = './output/results.csv',
            indep_graphs_path = './data/monomer_graphs',
            data_split = 0,
            save=False):

    convo = 'with' if convolutional else 'without'
    print(f"now training model with:\n -{layers} layer(s),\n -{convo} convolutional end layer,\n -lr={lr},\n -{epochs} training epochs\n -on data split {data_split}")
    # Choose the data, features and targets to use and GET THE DATA GOING
    node_features = ['nt_code', 'sse', 'module']
    node_target = ['binding_small-molecule']

    # .json --> networkx graphs annotated with x3DNA-DSSR and BayesPairing2
    json_graphs = os.listdir(json_graphs_path)
    print(json_graphs)
    try:  json_graphs.remove('.DS_Store')
    except: pass

    if data_split == 0:
        test_split = 0.8
        num_test_graphs = int( len(json_graphs) * (1 - test_split ) )
        #getting train and test indices
        test_indices = rand.sample( range( len( json_graphs ) ), num_test_graphs )
        train_indices = np.setdiff1d( range( len( json_graphs ) ), test_indices )
        # getting train and test graphs
        test_graphs = [json_graphs[i] for i in test_indices]
        train_graphs = [json_graphs[i] for i in train_indices]
    elif data_split == 1:
        indep_chains = os.listdir( indep_graphs_path)
        #removing the intersection of both graph sets
        json_graphs = list( np.setdiff1d( json_graphs, indep_chains ) )
        #adding half of the independent chains to the training set...
        indep_chains_train_indices = rand.sample( range( len( indep_chains) ), int( len(indep_chains) /2. ) )
        indep_chains_train_graphs = [indep_chains[i] for i in indep_chains_train_indices ]
        for chain in indep_chains_train_graphs:
            json_graphs.append( chain )
        train_graphs = json_graphs

        #getting test graphs from independent chains
        test_indices = rand.sample( range( len( indep_chains ) ), len(indep_chains) - len(indep_chains_train_indices) )
        print( "split correct?", len(indep_chains) == len(indep_chains_train_indices) + len(indep_chains_train_indices) )
        test_graphs = [indep_chains[i] for i in test_indices]
    elif data_split == 2:
        indep_chains = os.listdir( indep_graphs_path )
        # removing the intersection of both graph sets
        json_graphs = list(np.setdiff1d(json_graphs, indep_chains))

        #setting up train and test sets from multi-chain rnas
        test_split = 0.8
        num_test_graphs = int( len(json_graphs) * (1 - test_split ) )
        #getting train and test indices
        test_indices = rand.sample( range( len( json_graphs ) ), num_test_graphs )
        train_indices = np.setdiff1d( range( len( json_graphs ) ), test_indices )
        # getting train and test graphs
        test_graphs = [json_graphs[i] for i in test_indices]
        train_graphs = [json_graphs[i] for i in train_indices]

        #setting up train and test sets from multi-chain rnas
        indep_chains_train_indices = rand.sample( range( len( indep_chains) ), int( len(indep_chains) /2. ) )
        indep_chains_train_graphs = [indep_chains[i] for i in indep_chains_train_indices ]
        for chain in indep_chains_train_graphs:
            train_graphs.append( chain )
        easy_test_indices = rand.sample( range( len( indep_chains ) ), len(indep_chains) - len(indep_chains_train_indices) )
        easy_test_graphs = [indep_chains[i] for i in easy_test_indices]
        easy_test_dataset = GraphDataset(data_path=json_graphs_path,
                                    all_graphs=easy_test_graphs,
                                    node_features=node_features,
                                    node_target=node_target,
                                    node_simfunc=None,
                                    annotated=True)
        easy_test_loader = GraphLoader(dataset=easy_test_dataset, split=False).get_data()
    else:
        test_graphs = [json_graphs[i] for i in rand.sample( range( len( json_graphs ) ), int( len(json_graphs) * 0.2 ) )]
        train_graphs = [json_graphs[i] for i in np.setdiff1d( range( len( json_graphs ) ), rand.sample( range( len( json_graphs ) ), int( len(json_graphs) * 0.2 ) ) )]


    #loading custom training dataset
    train_dataset = GraphDataset(data_path=json_graphs_path,
                                 all_graphs=train_graphs,
                                 node_features=node_features,
                                 node_target=node_target,
                                 node_simfunc=None,
                                 annotated=True)
    train_loader = GraphLoader(dataset=train_dataset, split=False).get_data()
    test_dataset = GraphDataset(data_path=json_graphs_path,
                                all_graphs=test_graphs,
                                node_features=node_features,
                                node_target=node_target,
                                node_simfunc=None,
                                annotated=True)
    test_loader = GraphLoader(dataset=test_dataset, split=False).get_data()


    #defining variables for dgl classifier and embedder
    input_dim, target_dim = train_dataset.input_dim, train_dataset.output_dim
    # print('input_dim', input_dim, "--- output_dim", target_dim)

    #creating models
    embedder_model = models.Embedder(dims=[10, 10], infeatures_dim=input_dim)

    classifier_model = models.Classifier(embedder=embedder_model,
                                         classif_dims=[target_dim],
                                         conv_output=convolutional,
                                         num_layers=layers,
                                         verbose=True)

    # Finally, get the training going
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=lr)
    learn.train_supervised(model=classifier_model,
                           optimizer=optimizer,
                           train_loader=train_loader,
                           learning_routine=LearningRoutine(num_epochs=epochs) )

    ##compute loss
    def compute_loss(classifier, test_loader):
        classifier.eval()
        device = classifier.current_device
        true, predicted = list(), list()
        #iterate through the test loader
        for batch_idx, (graph, graph_sizes) in enumerate(test_loader):
            # Get data on the devices
            graph = send_graph_to_device(graph, device)

            # Do the computations for the forward pass
            with torch.no_grad():
                labels = graph.ndata['target']
                out = classifier(graph)

                true.append(misc.tonumpy(labels))
                predicted.append(misc.tonumpy(out))
        true = np.concatenate(true)
        predicted = np.concatenate(predicted)
        loss = roc_auc_score(true, predicted)
        return loss, true, predicted


    #compute accuracy
    def compute_accuracy(true, predicted):
        binarized = []
        stats = {'true_pos': 0, 'true_neg': 0, 'false_pos': 0, 'false_neg': 0}

        for i in range( len( predicted ) ):
            temp = []
            for j in range( len( predicted[i] ) ):
                val = 0 if predicted[i][j] < 0.2 else 1
                temp.append( val )
            binarized.append( temp )

        for i in range(len(true)):
            for j in range(len(true[i])):
                if true[i][j] == binarized[i][j]:
                    if true[i][j] == 1:
                        stats['true_pos'] += 1
                    else:
                        stats['true_neg'] += 1
                else:
                    if true[i][j] == 1:
                        stats['false_neg'] += 1
                    else:
                        stats['false_pos'] += 1
        return stats

    #evaluate model
    loss, true, predicted = compute_loss(classifier_model, test_loader)
    stats = compute_accuracy(true, predicted)

    print("Statistics on test set:\n",stats)
    sensitivity = stats['true_pos'] / (stats['true_pos'] + stats['false_neg'])
    specificity = stats['true_neg'] / (stats['false_pos'] + stats['true_neg'])
    accuracy = (stats['true_pos'] + stats['true_neg']) / (stats['true_pos'] + stats['true_neg'] + stats['false_neg'] + stats['false_pos'])
    print("\tSensitivity:", sensitivity)
    print("\tSpecifity:", specificity)
    print("\tAccuracy:", accuracy)
    print("\tLoss =",loss)

    # if the test set is split in 2 (i.e. for data_split =2 ), evaluate model on second test set
    if data_split == 2:
        easy_loss, easy_true, easy_predicted = compute_loss(classifier_model, easy_test_loader)
        easy_stats = compute_accuracy(easy_true, easy_predicted)
        print("DATA_SPLIT = 2:\nStatistics on easy test set:\n",easy_stats)
        easy_sensitivity = easy_stats['true_pos'] / (easy_stats['true_pos'] + easy_stats['false_neg'])
        easy_specificity = easy_stats['true_neg'] / (easy_stats['false_pos'] + easy_stats['true_neg'])
        easy_accuracy = (easy_stats['true_pos'] + easy_stats['true_neg']) / (easy_stats['true_pos'] + easy_stats['true_neg'] + easy_stats['false_neg'] + easy_stats['false_pos'])
        print("\tSensitivity:", easy_sensitivity)
        print("\tSpecifity:", easy_specificity)
        print("\tAccuracy:", easy_accuracy)
        print("\tLoss =",easy_loss)

    #write results to csv file
    param_names = ['>', 'convolutional_model', 'layers', 'lr', 'epochs', 'data_split', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'Sensitivity', 'Specificity', 'Accuracy']
    params = [ '?', convolutional, layers, lr, epochs, data_split, stats['true_pos'], stats['true_neg'], stats['false_pos'], stats['false_neg'], sensitivity, specificity, accuracy]
    with open(csv_path, 'a') as f:
        csvwriter = writer(f)
        csvwriter.writerow(param_names)
        csvwriter.writerow(params)
        if data_split==2:
            easy_params = [ '?', '^', '^', '^', '^', '^', easy_stats['true_pos'], easy_stats['true_neg'], easy_stats['false_pos'], easy_stats['false_neg'], easy_sensitivity, easy_specificity, easy_accuracy]
            csvwriter.writerow(easy_params)

    # if you want to save the model
    if save:
        model_name = "./models/gnn_weights.pt"
        # model_name = f"./models/gnn_weights_.pt"
        torch.save(classifier_model, model_name)
        print(f"Model successfully saved to {model_name}.")
    print("Done.")

if __name__ == '__main__':
    # get command line arguments
    graph_path = sys.argv[1]
    convs = sys.argv[2]
    layer = sys.argv[3]
    epoch = sys.argv[4]
    ds = sys.argv[5]
    save = sys.argv[6]

    # train gnn model
    run_gnn('./data/15cutoff_multimer_graphs',
            convolutional=bool(convs),
            layers=int(layer),
            epochs=int(epoch),
            data_split=int(ds),
            save=save
            )

    # hyper-parameter grid search:
    # convolutionals = [False, True]
    # layers = [1, 2, 3, 4]
    # lrs = [0.01, 0.001, 0.0001]
    # epochs = [10, 25,  50]
    # data_splits = [0, 1, 2]

    # for convs in convolutionals:
    #     for layer in layers:
    #         for lr in lrs:
    #             for epoch in epochs:
    #                 for ds in data_splits:
    #                     run_gnn(convolutional = convs, layers = layer, lr=lr,epochs=epoch,data_split=ds)
