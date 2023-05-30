python util/data_preprocess.py --mode data_process --bert_model_path ../bert-base-uncased/
python util/data_preprocess.py --mode gen_qd --bert_model_path ../bert-base-uncased/
python util/T5_data_preprocess.py --mode extract_doc_content --bert_model_path ../bert-base-uncased/
python util/T5_data_preprocess.py --mode gen_data --bert_model_path ../bert-base-uncased/
python util/seq_data_preprocess.py --bert_model_path ../bert-base-uncased/ --bert_emb_len 256

python run.py --mode cl_T5 --fold 1
python run.py --mode cl_T5 --fold 2
python run.py --mode cl_T5 --fold 3
python run.py --mode cl_T5 --fold 4
python run.py --mode cl_T5 --fold 5

CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --temperature_cll 0.8 --mode cl_list --fold 1 > nohup1.out 2>&1&&
CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --temperature_cll 0.8 --mode cl_list --fold 2 > nohup2.out 2>&1&&
CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --temperature_cll 0.4 --mode cl_list --fold 3 > nohup3.out 2>&1&&
CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --temperature_cll 0.7 --mode cl_list --fold 4 > nohup4.out 2>&1&&
CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --temperature_cll 0.8 --mode cl_list --fold 5 > nohup5.out 2>&1&

CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --temperature_cll 0.8 --mode cl_div --fold 1 > nohup1.out 2>&1&&
CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --temperature_cll 0.8 --mode cl_div --fold 2 > nohup2.out 2>&1&&
CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --temperature_cll 0.4 --mode cl_div --fold 3 > nohup3.out 2>&1&&
CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --temperature_cll 0.7 --mode cl_div --fold 4 > nohup4.out 2>&1&&
CUDA_VISIBLE_DEVICES=1 nohup python -u run.py --temperature_cll 0.8 --mode cl_div --fold 5 > nohup5.out 2>&1&