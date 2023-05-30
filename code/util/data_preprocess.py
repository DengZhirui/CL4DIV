import sys, os, argparse
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import multiprocessing
import gzip, copy
import xml.dom.minidom
from util.div_type import *
from util.utils import set_seed, split_list
set_seed()
MAXDOC = 50
REL_LEN = 18


def data_process():
    ''' get subtopics for each query '''
    # qd[qid] = class(qid, query, subtopic_id_list, [class(subtopic_id, subtopic), ...])
    qd = get_query_dict()
    ''' get documents dictionary '''
    dd, ds = get_docs_dict()
    ''' get diversity judge for documents '''
    qd = get_doc_judge(qd, dd, ds)
    ''' get the stand best alpha-nDCG from DSSA '''
    get_stand_best_metric(qd)
    ''' get the best ranking for top n relevant documents and save as files'''
    calculate_best_rank(qd)


def get_query_dict():
    dq_dict = {}
    topics_list = []
    for year in ['2009','2010','2011','2012']:
        filename = '../data/clueweb_data/wt_topics/wt' + year + '.topics.xml'
        DOMTree = xml.dom.minidom.parse(filename)
        collection = DOMTree.documentElement
        topics = collection.getElementsByTagName("topic")
        topics_list.extend(topics)
    ''' load subtopics for each query '''
    for topic in topics_list:
        if topic.hasAttribute("number"):
            qid = topic.getAttribute("number")
        query = topic.getElementsByTagName('query')[0].childNodes[0].data
        subtopics = topic.getElementsByTagName('subtopic')
        subtopic_id_list = []
        subtopic_list = []
        for subtopic in subtopics:
            if subtopic.hasAttribute('number'):
                subtopic_id = subtopic.getAttribute('number')
                subtopic_id_list.append(subtopic_id)
            sub_query = subtopic.childNodes[0].data
            subtopic_list.append(sub_query)
        dq = div_query(qid, query, subtopic_id_list, subtopic_list)
        dq_dict[str(qid)] = dq
    return dq_dict


def get_docs_dict():
    '''
    get the relevance score of the documents
    docs_dict[qid] = [doc_id, ...]
    docs_rel_score_dict[qid] = [score, ...]
    '''
    docs_dict = {}
    docs_rel_score_dict = {}
    filename = '../data/baseline_data/ideal_ranking.txt'
    f = open(filename)
    for line in f:
        qid, _, docid, _, score, _ = line.split(' ')  # score is the retrieval stage score, calculated by BM25 etc.
        if str(qid) not in docs_dict:
            docs_dict[str(qid)] = []
            docs_rel_score_dict[str(qid)] = []
        docs_dict[str(qid)].append(str(docid))
        docs_rel_score_dict[str(qid)].append(float(score))
    ''' Normalize the relevance score of the documents '''
    for qid in docs_rel_score_dict:
        temp_score_list = copy.deepcopy(docs_rel_score_dict[qid])
        for i in range(len(temp_score_list)):
            temp_score_list[i] = docs_rel_score_dict[qid][0]/docs_rel_score_dict[qid][i]
        docs_rel_score_dict[qid] = temp_score_list
    return docs_dict, docs_rel_score_dict


def get_doc_judge(qd, dd, ds):
    '''
    load document list and relevance socre list for the corresponding query
    qd : query dictionary
    dd : document dictionary
    ds : document relevance score dictionary
    '''
    get_query_suggestion(qd)
    for key in qd:
        qd[key].add_docs(dd[key])
        qd[key].add_docs_rel_score(ds[key])
    filename = '../data/baseline_data/whole.qrels'
    f = open(filename, 'r')
    for line in f:
        qid, subtopic, docid, judge = line.split(' ')
        judge = int(judge)
        if judge > 0:
            if str(docid) in qd[str(qid)].subtopic_df.index.values:
                qd[str(qid)].subtopic_df[str(subtopic)][str(docid)] = 1
    return qd


def get_query_suggestion(dq):
    dq_dict = {}
    filename = '../data/baseline_data/query_suggestion.xml'
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    topics = collection.getElementsByTagName("topic")
    ''' load subtopics for each query '''
    for topic in topics:
        if topic.hasAttribute("number"):
            qid = topic.getAttribute("number")
        query = topic.getElementsByTagName('query')[0].childNodes[0].data
        subtopics = topic.getElementsByTagName('subtopic1')
        subtopic_id_list = []
        subtopic_list = []
        for subtopic in subtopics:
            if subtopic.hasAttribute('number'):
                subtopic_id = subtopic.getAttribute('number')
                subtopic_id_list.append(subtopic_id)
            suggestion = subtopic.getElementsByTagName('suggestion')[0].childNodes[0].data
            subtopic_list.append(suggestion)
        dq[str(qid)].add_query_suggestion(subtopic_list)
    return dq_dict


def get_stand_best_metric(qd):
    ''' load best alpha-nDCG from DSSA '''
    std_dict = pickle.load(open('../data/baseline_data/stand_metrics.data', 'rb'))
    for qid in std_dict:
        m = std_dict[qid]
        target_q = qd[str(qid)]
        target_q.set_std_metric(m)


def calculate_best_rank(qd):
    data_dir = '../data/attn_data/best_rank/'
    if not os.path.exists('../data/attn_data/'):
        os.makedirs('../data/attn_data/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    q_list = []
    for key in qd:
        x = copy.deepcopy(qd[key])
        q_list.append((str(key), x))
    jobs = []
    task_list = split_list(q_list, 8)
    for task in task_list:
        p = multiprocessing.Process(target = data_process_worker, args = (task, ))
        jobs.append(p)
        p.start()


def data_process_worker(task):
    for item in task:
        qid = item[0]
        dq = item[1]
        ''' get the best ranking for the top 50 relevant documents '''
        dq.get_best_rank(MAXDOC)
        pickle.dump(dq, open('../data/attn_data/best_rank/'+str(qid)+'.data', 'wb'), True)


def generate_qd():
    ''' generate diversity_query file from data_dir '''
    data_dir = '../data/attn_data/best_rank/'
    files = os.listdir(data_dir)
    files.sort(key = lambda x:int(x[:-5]))
    query_dict = {}
    for f in files:
        file_path = os.path.join(data_dir, f)
        temp_q = pickle.load(open(file_path, 'rb'))
        query_dict[str(f[:-5])] = temp_q
    pickle.dump(query_dict, open('../data/attn_data/div_query.data', 'wb'), True)
    return query_dict


def intent_coverage_data_prepare():
    intent_coverage = {}
    qd = pickle.load(open('../data/attn_data/div_query.data', 'rb'))
    for qid in tqdm(qd):
        intent_coverage[qid] = []
        docs = qd[qid].best_docs_rank
        for i in range(len(docs)-1):
            for j in range(i+1, len(docs)):
                intent_i = list(qd[str(qid)].subtopic_df.loc[docs[i]])
                intent_j = list(qd[str(qid)].subtopic_df.loc[docs[j]])
                if intent_i == intent_j and (tuple((docs[i], docs[j])) not in intent_coverage[qid]) and \
                        (tuple((docs[i], docs[j]))) not in intent_coverage[qid]:
                    intent_coverage[qid].append(tuple((docs[i], docs[j])))
    with gzip.open('../data/attn_data/intent_coverage.pkl.gz', 'wb') as f:
        pickle.dump(intent_coverage, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', type=str, default="data_process", help="run mode")
    parser.add_argument("--bert_model_path", default="../bert-base-uncased/", type=str, help="")
    args = parser.parse_args()
    
    if args.mode == 'data_process':
        '''get q-d relevance, q-subtopic relevance, q and subtopic's rel and doc2vec features'''
        data_process()
    elif args.mode == 'gen_qd':
        ''' generate query data files : div_query.data'''
        generate_qd()
        ''' generate Training Datasets : listpair_train.data '''
        D = div_dataset()
        D.get_listpair_train_data()
        intent_coverage_data_prepare()    


