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

### Training the contrastive learning model. 
```
python run.py --mode cl_T5 --fold 1
python run.py --mode cl_list --fold 1
```

### Diversified Ranking Stage
```
python run.py --mode cl_div --fold 1
```

### Others
The dataset and models will be released after the paper is accepted. 
