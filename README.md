# CL4DIV
Source code of paper ```CL4DIV: A Contrastive Learning Framework for Search Result Diversification```

## Quick Start
### Prepare for the training data. 
```
python util/data_preprocess.py --mode data_process --bert_model_path ../bert-base-uncased/
python util/data_preprocess.py --mode gen_qd --bert_model_path ../bert-base-uncased/
python util/T5_data_preprocess.py --mode extract_doc_content --bert_model_path ../bert-base-uncased/
python util/T5_data_preprocess.py --mode gen_data --bert_model_path ../bert-base-uncased/
python util/seq_data_preprocess.py --bert_model_path ../bert-base-uncased/ --bert_emb_len 256
```

### Training the contrastive learning tasks. 
```
python run.py --mode cl_T5 --fold 1
python run.py --mode cl_list --fold 1
```

### Diversified Ranking Stage
```
python run.py --mode cl_div --fold 1
```

## Visualization
To investigate the effect of contrastive learning on data representations, we visualize the document representation distribution before and after contrastive learning. We randomly select a query from the dataset and map the document representations to 2-dimensional vectors via principal component analysis (PCA). The results are illustrated in **Figure 1**, where documents with the same subtopics are marked in the same color. 

![image text](https://github.com/DengZhirui/CL4DIV/blob/master/visual_query_61.png)

As shown in the left side, the initialized distribution of documents with the same subtopics is more scattered, whereas documents containing different subtopics are mixed together. After contrastive learning, we notice two changes: 

* Documents covering the same subtopics are clustered together. For query \#61, documents covering the 3rd subtopic (in blue) become closer after contrastive learning. 

* Documents with different subtopics are separated. Documents containing subtopics 3 and 4 (in blue and purple) move far apart. 

## Citations
If you use the code, please cite the following paper:

```
@inproceedings{Deng,
  author    = {Zhirui Deng and
               Zhicheng Dou and
               Yutao Zhu and
               Ji-Rong Wen},
  title     = {CL4DIV: A Contrastive Learning Framework for Search Result Diversification},
  url       = {https://doi.org/10.1145/3616855.3635851},
  doi       = {10.1145/3616855.3635851},
}

```
