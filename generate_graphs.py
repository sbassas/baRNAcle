import os
import json
from builtins import FileNotFoundError

import numpy as np
import networkx as nx

from networkx.readwrite import json_graph
from tqdm import tqdm

def do_all(in_path, out_path, cutoff):
    '''
    Use do_one method from rnaglib to convert an mmcif to a 2.5D graph using DSSR and RNAfold annotations
    on a directory of mmcif files. Many files will be created and deleted in the rnaglib_main directory
    during the process of running this command.

    Parameters
    ----------
    in_path: path to mmcif files
    out_path: dump path containing annotated graphs as in json link_node_graph format
    cutoff: distance cutoff in Å defining the region around a ligand considered a binding pocket

    Returns
    -------

    '''
    from lib.rnaglib_main.rnaglib.prepare_data.main import do_one
    dir = os.listdir(in_path)
    print("Converting mmcif files to 2.5D graphs...")
    for file in tqdm(dir):
        pdbid = file.split('.')[0]
        if not os.path.exists( os.path.join( out_path, pdbid + '.json' ) ):
            id, err = do_one( os.path.join( in_path, file ), out_path, cutoff= cutoff )

def load_json(filename):
    '''
    load a 2.5D graph stored in a json file

    Parameters
    ----------
    filename: path to json file containing 2.5D graph

    Returns
    -------
    out_graph: a 2.5D graph stored as a networkx object

    '''
    with open(filename, 'r') as f:
        js_graph = json.load(f)
    out_graph = json_graph.node_link_graph(js_graph)
    return out_graph


def bp_predict_all(dir_path, out_path='./lib/BayesPairing2/bayespairing/output', threshold=4, samples=1000, module_set='RELIABLE'):
    '''
    Predict recurrent RNA 3D module placements within a single chian at a time using parse_sequences
    module from BayesPairing2. Do this prediction step for a directory of 2.5D graphs representing both
    RNA monomers or multimers.

    Parameters
    ----------
    dir_path: path to 2.5D graph directory
    out_path: dump path for chain level prediction in .json format
    threshold: score threshold under which a module prediction is rejected as not confident enough
    samples: number of stochastic sampling tasks performed by BayesPairing2 per chain
    module_set: set of 3D modules BayesPairing2 is trained on prior to module predictions

    Returns
    -------

    '''
    jsons = os.listdir(dir_path)
    try: jsons.remove('.DS_Store')
    except: pass
    jsons.sort()
    print('Predicting recurrent 3D module locations for each individual chain...')

    seqs = []
    for file in tqdm(jsons):
        #load 2.5D graph
        graph = load_json(os.path.join( dir_path, file ))

        for chain in graph.graph['dbn']['single_chains'].keys():
            # formatting the base pair sequence string contained in 2.5D graph attributes to fit the
            # expected input criteria for BayesPairing2
            seq = graph.graph['dbn']['single_chains'][chain]['bseq'].upper().replace('&', '').replace('T','U')
            # select each chain in graph that falls in range [20; 1500]
            if 20 <= len( seq ) <= 1500:
                #predict using bp2
                out_name = file.split('.')[0] + '_' + chain
                if os.path.exists( os.path.join( out_path, out_name + '.json' ) ) :
                    print(out_name+ '.json', 'already exists.\n')
                else:
                    print('Predicting modules for', file, chain, 'chain...')
                    script_name = './lib/BayesPairing2/bayespairing/src/parse_sequences.py'
                    try:
                        command = f"python {script_name} -seq \'{seq}\' -samplesize {samples} -t {threshold} -d {module_set} -o {out_name}"
                        os.system(command)
                    except:
                        seqs.append(graph.graph['dbn']['single_chains'][chain]['bseq'])


def bp_annotate_one(path_2_2pt5d,
                    chain_name,
                    bp2_path,
                    dump_path,
                    threshold=4.5):
    '''
    Find the corresponding rnaglib 2.5D graph and BP2 module predictions for an input pdbid
    Annotate the 2.5D graph with 3D module information (location, score, and module id)

    Parameters
    ----------
    path_2_2pt5d: path to 2.5D graph
    chain_name: full name of chain of interest (including pdbid) in input 2.5D graph
    bp2_path: path to foler containing BayesPairing2 (BP2) output predictions
    dump_path: path where output subgraph should be dumped
    threshold: score value above which BP2 module predictions are selected

    Returns
    -------

    '''
    # open the 2.5D graph and the output from bp2
    pdbid = chain_name.split('_')[0]
    print("Annotating", pdbid, '...')
    try:
        graph = load_json(path_2_2pt5d)
    except FileNotFoundError:
        print("File not found for", path_2_2pt5d)
        return

    out = os.path.join(bp2_path, chain_name + '.json')
    f = open(out, 'r')
    bp2_out = json.load(f)
    # getting the predictions associated with each module, dict ( module_id: module_prediction)
    preds = bp2_out['all_hits']['input_seq']

    # getting a sub-dictionary of modules whose prediction score is above the threshold
    valid_modules = {k: v[0] for k, v in preds.items() if len(v) >= 1 and v[0][2] > float(threshold)}

    ##CREATE ANNOTATIONS
    annotations = {}
    print("Valid modules:")
    print("<id> -- <residues> -- <score> -- <node indices>")
    for id, module in list(valid_modules.items()):
        print(id, '--', module[0], '--', module[2], end='')
        if np.array(module[1]).shape[0] > 1:
            print('-- ', np.array([x for sub in module[1] for x in sub]))
            for x in [y for sub in module[1] for y in sub]:
                try:
                    if annotations[x] != None and module[2] >= annotations[x][0]:
                        annotations[x] = (module[2], id)
                except KeyError:
                    annotations[x] = (module[2], id)
        else:
            print('-- ', np.array(module[1]).ravel())
            for x in np.array(module[1]).ravel():
                try:
                    if annotations[x] != None and module[2] >= annotations[x][0]:
                        annotations[x] = (module[2], id)
                except KeyError:
                    annotations[x] = (module[2], id)

    ##ANNOTATING THE GRAPHS
    # defining variables
    matched = []
    subgraph = graph.__class__()

    # getting the identifier of the chain, which are the characters after the last underscore
    chain_id = chain_name.split('_')[-1]
    print("Chain_id = ", chain_id)
    # adding module id and module prediction score to a new 'module' attribute in each node
    for node in graph.nodes:
        # check if the node is in the right chain
        if graph.nodes[node]['chain_name'] == chain_id:
            # check if the node is in a module, -1 because of the different indexings
            if graph.nodes[node]['index_chain'] - 1 in annotations.keys():
                matched.append(graph.nodes[node]['index_chain'])
                graph.nodes[node]['module'] = annotations[graph.nodes[node]['index_chain'] - 1]
            else:
                graph.nodes[node]['module'] = None

    # checking there are no mismatches and incoherences in the annotation process
    print(len(matched), "nodes are part of 3D modules.")
    if len(matched) != len(annotations):
        print('mismatch between graph nodes and annotation indices')
        if len(matched) < 1:
            print("no matches, ERROR")

    # make subgraph for nodes in corresponding chain
    try:
        subgraph = graph.__class__()
        subgraph.add_nodes_from((n, graph.nodes[n])
                                for n in graph.nodes
                                if graph.nodes[n]['chain_name'] == chain_id)
        subgraph.add_edges_from((nt1, nt2, attr)
                                for nt1, neighbors in graph.adj.items()
                                if nt1 in subgraph.nodes
                                for nt2, attr in neighbors.items()
                                if nt2 in subgraph.nodes)
    except:
        print("Could not make subgraph for", chain_name)

    # save graph
    d = json_graph.node_link_data(subgraph)
    with open(os.path.join(dump_path, chain_name + '.json'), 'w') as f:
        json.dump(d, f)

    # visualization
    visualize = False #turned off because even more verbose
    if visualize:
        for node in graph.nodes:
            if graph.nodes[node]['chain_name'] == chain_id:
                print(graph.nodes[node]['nt_id'], graph.nodes[node]['index_chain'], graph.nodes[node]['module'])


def bp_annotate_all(graph_path, bp_path, dump_path, threshold=4.5):
    '''
    Run bp_annotate_one for a whole set of 2.5D graphs and their corresponding BayesPairing2 outputs.
    Since graph_path contains RNA monomer and mutlimer graphs but bp_path contains only single chain
    predictions, this method also takes care of subgraphing the 2.5D graphs with only the nodes and
    edges that belong to the chain in the prediction file.

    Parameters
    ----------
    graph_path: path to 2.5D graphs directory
    bp_path: path to BayesPairing2 output predictions directory
    dump_path: dump path of 2.5D subgraphs annotated with module predicitons
    threshold: threshold score for module selection during annotation

    Returns
    -------

    '''
    predictions = os.listdir(bp_path)
    predictions = [x for x in predictions if x.endswith('.json')]
    chain_names = [x.split('.')[0] for x in predictions]
    print("Annotating 2.5D graphs of individual RNA chains with their corresponding 3D module predictions...")
    for i in tqdm(range(len(chain_names))):
        print(chain_names[i].split('_')[0], '.json  ---', chain_names[i])
        bp_annotate_one(os.path.join(graph_path, chain_names[i].split('_')[0] + '.json'),
                        chain_names[i],
                        bp_path,
                        dump_path=dump_path,
                        threshold=threshold
                        )


def fix_node_modules(dir_path):
    '''
    Giving each node attribute in each graph in input directory a key value pair 'module': None to avoid
    any KeyError Exceptions down the road

    Parameters
    ----------
    dir_path: path to 2.5D graph directory

    Returns
    -------

    '''
    dir = os.listdir(dir_path)
    print("Fixing node attribute 'model' for 2.5D graph annotation...")
    for file in tqdm(dir):
        graph = load_json( os.path.join( dir_path, file ) )
        try:
            for node in graph.nodes:
                if graph.nodes[node]['module'] != None:
                    continue
        except:
            print(file, "did not have any module annotations.")
            for node in graph.nodes:
                graph.nodes[node]['module'] = None
        finally:
            d = nx.readwrite.json_graph.node_link_data(graph)
            with open( dir_path, 'w' ) as f:
                json.dump(d, f)

def fix_node_sses(dir_path):
    '''
    Fixing a bug where each sse node attribute is another dictionary. This method removes that extra layers from
    each node in each graph in the input layer.

    Parameters
    ----------
    dir_path: path to 2.5D graph directory

    Returns
    -------

    '''
    print("Fixing node attribute 'sse' for 2.5D graph annotation...")
    dir = os.listdir(dir_path)
    for file in tqdm(dir):
        g = load_json( os.path.join( dir_path, file ) )
        for node in g.nodes:
            temp = g.nodes[node]['sse']['sse']
            g.nodes[node]['sse'] = temp
        #write graph to output directory
        d = nx.readwrite.json_graph.node_link_data(g)
        with open( os.path.join( dir_path,file ), 'w') as f:
            json.dump(d, f)

def get_independent_rnas(graph_path, out_path):
    '''
    Get a list of RNA monomer chains from a mixed list of 2.5D graph monomers and multimers

    Parameters
    ----------
    graph_path: path to 2.5D graph directory
    out_path: dump path for monomer chains

    Returns
    -------

    '''
    indeps = []
    annot_indeps = []
    dir = os.listdir(graph_path)
    for file in dir:
        graph = load_json( os.path.join(graph_path,file))
        chain_names = list( graph.graph['dbn']['single_chains'].keys() )
        if len( list( graph.graph['dbn']['single_chains'].keys() ) ) == 1 and 20 <= len (graph.graph['dbn']['single_chains'][chain_names[0]]['bseq']) <= 1500:
            indeps.append(file[:4])
            sse = False
            smb = False
            for k,v in graph.nodes.items():
                if v['binding_small-molecule'] != None:
                    smb = True
                if v['sse']['sse'] != None:
                    sse = True
            if sse and smb:
                annot_indeps.append( file.split('.')[0] )
    for pdbid in indeps:
        os.system(f"cp {os.path.join(graph_path, pdbid + '.json' )} {os.path.join(out_path, pdbid + '.json' )}")

def run_data_generation_pipeline(mmcif_dir, cutoff):
    #creating path variables
    RNAmer_graphs_dir = './temp_dir/RNAmer_graphs'
    RNAchain_graphs_dir = './temp_dir/RNAchain_graphs'
    monomer_graphs_dir = './temp_dir/monomer_graphs'
    bp_predictions_dir = './lib/BayesPairing2/bayespairing/output'


    do_all(mmcif_dir,RNAmer_graphs_dir,cutoff=cutoff)
    bp_predict_all(RNAmer_graphs_dir)
    bp_annotate_all(RNAmer_graphs_dir, bp_predictions_dir, RNAchain_graphs_dir)
    fix_node_modules(RNAchain_graphs_dir)
    fix_node_sses(RNAchain_graphs_dir)
    get_independent_rnas(RNAmer_graphs_dir, monomer_graphs_dir)

    #removing the contents of the intermediate graph directory
    os.system(f"rm {RNAmer_graphs_dir}/*")

    print("\nDone.\n")

if __name__ == '__main__':
    #run_data_generation_pipeline('./data/15cutoff_multimer_graphs', 15)
    run_data_generation_pipeline(sys.argv[1], sys.argv[2])