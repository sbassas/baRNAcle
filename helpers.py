import json
import os
from networkx.readwrite import json_graph


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


def get_valid_sequences(path):
    # flag to include node info or not
    sse_smb = True

    dir = os.listdir(path)
    dir.sort()

    annotated_seqs = [] if sse_smb else None
    valid_seqs = []

    # method to parse through the chains of graphs
    for file in dir:
        # load graph
        file_name = os.path.join(path, file)
        # print(file_name)
        try:
            graph = load_json(file_name)
            # get number of unique nucleic acid chains in graph
            chain_names = list(graph.graph['dbn']['single_chains'].keys())
            chain_nums = len(chain_names)
            # get sequence for each chain
            for i in range(chain_nums):
                if 20 <= len(graph.graph['dbn']['single_chains'][chain_names[i]]['bseq']) <= 1500:
                    valid_seqs.append((file.split(',')[0] + '_' + chain_names[i],
                                       graph.graph['dbn']['single_chains'][chain_names[i]]['bseq']))
                    if sse_smb:
                        # identifying the chains with sse and smb annots
                        chain_id = chain_names[i].split('_')[-1]
                        sse = False
                        smb = False
                        for k, v in graph.nodes.items():
                            if k.split('.')[1] == chain_id:
                                if v['binding_small-molecule'] != None:
                                    smb = True
                                if v['sse']['sse'] != None:
                                    sse = True
                        if sse and smb:
                            annotated_seqs.append(file.split(',')[0] + '_' + chain_names[i])
        except:
            pass
    return valid_seqs, annotated_seqs

def get_whole_rnas_with_annots(path):
    dir = os.listdir(path)
    dir.sort()
    annotated_seqs = []
    for file in dir:
        graph = load_json( os.path.join(path, file))
        sse = False
        smb = False
        for k,v in graph.nodes.items():
            if v['binding_small-molecule'] != None:
                smb = True
            if v['sse']['sse'] != None:
                sse = True
        if sse and smb:
            annotated_seqs.append( file[:4] )
    return annotated_seqs