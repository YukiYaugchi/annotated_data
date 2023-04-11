import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
import wave
import sys
from dotmap import DotMap
from tqdm import tqdm

sys.path.append('../')
import espnet

# from espnet2.bin.asr_inference_ import Speech2Text 
# from espnet2.bin.asr_bifurcation_transducer_inference import Speech2Text
from espnet2.bin.asr_parallel_transducer_inference import Speech2Text

from src.datasets.dataset_asr_inference2_inverse import get_dataloader, get_dataset
from src.utils.utils import load_config

MLA=True


# user : sakuma
# OUTDIR = '/mnt/aoni04/jsakuma/data/ATR_Annotated/data_-500_2000/texts/cbs-t_mla_bifurcation'
# asr_base_path = "/mnt/aoni04/jsakuma/development/espnet-peter/egs2/atr/asr1"


# user : parallel
"""
OUTDIR = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator/data/texts/cbs-t_mla_parallel'
asr_base_path = "/mnt/aoni04/yaguchi/code/espnet/egs2/atr/asr1"
asr_train_config = "exp/asr_train_asr_cbs_transducer_848_finetune_raw_jp_char_sp/config.yaml"
asr_model_file = "exp/asr_train_asr_cbs_transducer_848_finetune_raw_jp_char_sp/valid.loss_transducer.ave_10best.pth"
"""
OUTDIR = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator/data/texts/test'
asr_base_path = "/mnt/aoni04/yaguchi/code/test/espnet/egs2/atr/asr1"
asr_train_config = "exp/asr_train_asr_parallel_cbs_transducer_raw_jp_char_sp/config.yaml"
asr_model_file = "exp/asr_train_asr_parallel_cbs_transducer_raw_jp_char_sp/valid.loss_transducer.ave_1best.pth"


# user : bifurcation
"""
OUTDIR='/mnt/aoni04/yaguchi/code/ResponseTimingEstimator/data/texts/cbs-t_mla_bifurcation'
asr_base_path="/mnt/aoni04/yaguchi/code/espnet/egs2/atr/asr1"
asr_train_config = "exp/asr_train_asr_bifurcation_cbs_transducer_848_raw_jp_char_sp/config.yaml"
asr_model_file = "exp/asr_train_asr_bifurcation_cbs_transducer_848_raw_jp_char_sp/valid.loss_transducer.ave_10best.pth"
"""


speech2text = Speech2Text(
    asr_base_path=asr_base_path,
    asr_train_config=os.path.join(asr_base_path, asr_train_config),
    asr_model_file=os.path.join(asr_base_path, asr_model_file),
    token_type=None,
    bpemodel=None,
    beam_size=5,
    beam_search_config={"search_type": "maes"},
    lm_weight=0.0,
    nbest=1,
    #device = "cuda:0", # "cpu",
    device = "cpu",
)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_recognize_frame(data):
    output_list = []
    speech = data.astype(np.float16)/32767.0 #32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
    sim_chunk_length = 2048 # 640
       
    if len(speech) <= speech2text._raw_ctx*4:
        add = speech2text._raw_ctx*4 - len(speech) + 1
        pad = np.zeros(speech2text._raw_ctx*1+add)
    else:
        pad = np.zeros(speech2text._raw_ctx*1)
        
    speech = np.concatenate([speech, pad])
    speech2text.reset_inference_cache()
    
    if sim_chunk_length > 0:
        for i in range(len(speech)//sim_chunk_length):
            hyps = speech2text.streaming_decode(speech=speech[i*sim_chunk_length:(i+1)*sim_chunk_length], is_final=False)
            if not MLA:
                results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps))
            else:
                results = speech2text.hypotheses_to_results(speech2text.beam_search.sort_nbest(hyps[1]))
                        
            if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                output_list.append(text)
            else:
                output_list.append("")
        
    return output_list
    

def run(args):
    config = load_config(args.config)
    seed_everything(config.seed)
    #if args.gpuid >= 0:
    config.gpu_device = args.gpuid
        
    if config.gpu_device>=0:
        device = torch.device('cuda:{}'.format(config.gpu_device))
    else:
        device = torch.device('cpu')
    
    dataset = get_dataset(config)      
    loader = get_dataloader(dataset, config)
        
    for j, batch in enumerate(tqdm(loader)):

        chs = batch[0]           
        wavs = batch[5]#.to(device)
        wav_lengths = batch[6] #.to(self.device)
        wav_paths = batch[7] #.to(self.device)
        batch_size = int(len(chs))

        for i in range(batch_size):
            name_part = wav_paths[i].split('/')[-1].split('_')
            dir_name = '_'.join(name_part[:-1])
            file_name = '_'.join(name_part).replace('.wav', '')

            os.makedirs(os.path.join(OUTDIR, '{}'.format(dir_name)), exist_ok=True)
            df_path = os.path.join(OUTDIR, '{}/{}.csv'.format(dir_name, file_name))

            if os.path.isfile(df_path):
                continue 

            wav = wavs[i][:wav_lengths[i]].detach().cpu().numpy()

            text = get_recognize_frame(wav)

            df = pd.DataFrame({'asr_recog': text})
            df.to_csv(df_path, encoding='utf-8', index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs/preprocess.json', help='path to config file')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu device id')
    args = parser.parse_args()
    run(args)