import os
import glob
import json
import wave
import struct
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random

from tqdm import tqdm


name_mapping = {'F1(伊藤)': 'F1',
                'F2(不明)': 'F2',
                'F3(中川)': 'F3',
                'F4(川村)': 'F4',
                'M1(黒河内)': 'M1',
                'M2(平林)': 'M2',
                'M3(浜田)': 'M3',
                'M4(不明)': 'M4'
               }

MAX_LEN = 15000000

# 直前の発話のみ
# 出力: CNN-AE feature, VAD出力ラベル, 最後のIPU=1ラベル
class ATRDataset(Dataset):
    def __init__(self, config, cv_id, split='train', subsets=['M1_all'], speaker_list=None):
        self.config = config
        self.data_dir = self.config.data_params.data_dir
                              
        self.file_names = []
        # alldata or cross validation
        if cv_id == -1:
            name_path = os.path.join(self.data_dir, 'names/M1_{}.txt'.format(split))
            with open(name_path) as f:
                    lines = f.readlines()
            self.file_names = [line.replace('\n', '') for line in lines]
        else:
            for sub in subsets:
                name_path = os.path.join(self.data_dir, "names/{}.txt".format(sub))
                with open(name_path) as f:
                    lines = f.readlines()
                self.file_names += [line.replace('\n', '') for line in lines]
              
        # da_list
        with open(self.config.data_params.da_vocab_path) as f:
            da_vocab = f.read().split("\n")
            if da_vocab[-1] == '':
                da_vocab = da_vocab[:-1]
        self.da_vocab = da_vocab
        self.da_num = self.config.data_params.da_dim

        spk_file_path = os.path.join(self.data_dir, 'speaker_ids.csv')
        df_spk = pd.read_csv(spk_file_path, encoding="shift-jis")
        df_spk['operator'] = df_spk['オペレータ'].map(lambda x: name_mapping[x])
        filenames = df_spk['ファイル名'].to_list()
        spk_ids = df_spk['operator'].to_list()
        spk_dict  = dict(zip(filenames, spk_ids))
        if speaker_list is not None:
            self.file_names = [name for name in self.file_names if spk_dict[name+'.wav'] in speaker_list]
            
        path = self.config.data_params.token_list_path
        with open(path) as f:
            lines =f.readlines()
        self.tokens = [line.split()[0] for line in lines]
        
        self.frame_length = config.data_params.frame_size  # 1frame=50ms
        self.sample_rate = config.data_params.sampling_rate
        self.max_positive_length = config.data_params.max_positive_length # システム発話のターゲットの最大長(0/1の1の最大長) [frame]
        self.asr_delay = config.data_params.asr_decoder_delay # 実際のASRの遅延 [ms]        
        self.context_num = config.data_params.n_context
        self.max_timing = config.data_params.max_timing
        self.mim_timing = config.data_params.min_timing        
        self.text_dir = config.data_params.text_dir
        
        data = self.get_data()

        # alldata or cross validation
        if cv_id == -1:
            self.data = data
        else:
            random.seed(0)
            random.shuffle(data)
            
            NUM = len(data)//5
            sub1 = data[NUM*0:NUM*1]
            sub2 = data[NUM*1:NUM*2]
            sub3 = data[NUM*2:NUM*3]
            sub4 = data[NUM*3:NUM*4]
            sub5 = data[NUM*4:]
            
            if cv_id == 1:
                trainset = sub2+sub3+sub4+sub5
                testset = sub1
            elif cv_id == 2:
                trainset = sub3+sub4+sub5+sub1
                testset = sub2
            elif cv_id == 3:
                trainset = sub4+sub5+sub1+sub2
                testset = sub3
            elif cv_id == 4:
                trainset = sub5+sub1+sub2+sub3
                testset = sub4
            elif cv_id == 5:
                trainset = sub1+sub2+sub3+sub4
                testset = sub5
            else:
                NotImplemented
                
            if split == 'train':
                self.data = trainset
            else:
                self.data = testset
        
    def read_wav(self, wavpath):
        wf = wave.open(wavpath, 'r')

        # waveファイルが持つ性質を取得
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        fr = wf.getframerate()
        fn = wf.getnframes()

        x = wf.readframes(wf.getnframes()) #frameの読み込み
        x = np.frombuffer(x, dtype= "int16") #numpy.arrayに変換

        return x
    
    def token2idx(self, token, unk=1, maxlen=600): 
        if token != token or token == '':
            return [0]

        token = token.replace('<eou>', '')
        idxs = [self.tokens.index(t) if t in self.tokens else unk for t in token]

        return idxs

    def idx2token(self, idxs): 
        token = [self.tokens[idx] for idx in idxs]

        return token
    
    # da（シングルラベル）
    def da2label(self, da):
        label = [0 for _ in range(self.da_num)]
        if da in self.da_vocab:
            label[self.da_vocab.index(da)] = 1
        return label

    # das（マルチラベル）
    def das2labels(self, das):
        labels = [0 for _ in range(self.da_num)]
        das = das.split('/')
        for da in das:
            if da in self.da_vocab:
                labels[self.da_vocab.index(da)] = 1
        return labels

    def convert_frate(self, text, fr1=50, fr2=128):
        p50 = 0
        p128 = 0
        text50 = []
        text128=['']+text
        for i in range(len(text)*fr2//fr1):
            t = fr1*(i+1)
            p128 = t // fr2
            if len(text)-1<p128:
                text50.append(text[-1])
            else:
                text50.append(text[p128])

        return text50

    def get_last_ipu(self, turn):
        ipu_label = np.zeros(len(turn))
        sub = turn[1:]-turn[:-1]    
        if 1 in sub:
            idx = np.where(sub==1)[0][-1]
            ipu_label[idx+1:] = 1

        return ipu_label
    
    def get_turn_info(self, file_name):
        # 各種ファイルの読み込み
        df_turns_path = os.path.join(self.data_dir, 'csv/{}.csv'.format(file_name))
        df_vad_path = os.path.join(self.data_dir,'vad/{}.csv'.format(file_name))
        feat_list = os.path.join(self.data_dir, 'cnn_ae/{}/*_spec.npy'.format(file_name))
        wav_list = os.path.join(self.data_dir, 'wav/{}/*.wav'.format(file_name))
        wav_start_end_list = os.path.join(self.data_dir, 'wav_start_end/{}.csv'.format(file_name))
        feat_list = sorted(glob.glob(feat_list))
        wav_list = sorted(glob.glob(wav_list))
        
        df = pd.read_csv(df_turns_path)
        df_vad = pd.read_csv(df_vad_path)
        df_wav = pd.read_csv(wav_start_end_list)

        N = MAX_LEN//self.sample_rate*1000

        # vadの結果
        uttr_user = np.zeros(N//self.frame_length)
        uttr_agent = np.zeros(N//self.frame_length)      
        for i in range(len(df_vad)):
            spk = df_vad['spk'].iloc[i]
            start = (df_vad['start'].iloc[i]) // self.frame_length
            end = (df_vad['end'].iloc[i]) // self.frame_length

            if spk==1:
                uttr_user[start:end]=1
            else:
                uttr_agent[start:end]=1

        batch_list = []
        num_turn = len(df['spk'])
        
        for t in range(num_turn): 
            feat_path = feat_list[t]
            wav_path = wav_list[t]
            feat_file_name = feat_path.split('/')[-1].replace('.npy', '').replace('_spec', '')
            wav_file_name = wav_path.split('/')[-1].replace('.wav', '')
            
            assert feat_file_name == wav_file_name, "file name mismatch! check the feat-file and wav-file!"
            
            ch = df['spk'][t]
            offset = df['offset'][t]
            next_ch = df['nxt_spk'][t]
            wav_start = df_wav['wav_start'][t]//self.frame_length
            wav_end = df_wav['wav_end'][t]//self.frame_length
            cur_usr_uttr_end = df['end'][t]//self.frame_length
            timing = df['nxt_start'][t]//self.frame_length
            da = self.da2label(df['da1'][t])
            das = self.das2labels(df['da'][t])
            nxt_da = self.da2label(df['nxt_da1'][t])
            nxt_das = self.das2labels(df['nxt_da'][t])
            
            if df['nxt_start'][t] - df['offset'][t] == df['end'][t]:
                is_barge_in = False
            else:
                is_barge_in = True
            
            if wav_end - timing > self.max_positive_length:
                wav_end = timing + self.max_positive_length

            vad_user = uttr_user[wav_start:wav_end]

            turn_label = np.zeros(N//self.frame_length)
            turn_label[wav_start:cur_usr_uttr_end] = 1
            turn_label = turn_label[wav_start:wav_end]

            timing_target = np.zeros(N//self.frame_length)
            timing_target[timing:] = 1

            turn_timing_target = timing_target[wav_start:wav_end]

            last_ipu_user = self.get_last_ipu(vad_user)

            vad_label = vad_user
            last_ipu = last_ipu_user

            context = ''
            text_path = os.path.join(self.text_dir, f'{file_name}/{wav_file_name}.csv')
            df_text = pd.read_csv(text_path)
            df_text[pd.isna(df_text['asr_recog'])] = ''
            text = df_text['asr_recog'].tolist()
            text = [txt.replace('。', '')+context for txt in text]  
            
            text = self.convert_frate(text)
            
            # Delayの調節
            n = self.asr_delay//self.frame_length#-self.asr_delay2//self.frame_length
            if n >= 0:
                text = ['']*n+text
            else:
                text = text[abs(n):]+[text[-1]]*abs(n)
            
            kana = None
            
            idx = [self.token2idx(t) for t in text]
            
            length = len(vad_label)
            m = length-len(text)
            
            if m < 0:
                text = text[:m]
                idx = idx[:m]
            else:
                text = text+[text[-1]]*m
                idx = idx+[idx[-1]]*m
            
            if len(vad_label) == 0: 
                continue                       

            batch = {"ch": ch,
                     "offset": offset,
                     "is_barge_in": is_barge_in,
                     "text": text,
                     "kana": kana,
                     "idx": idx,
                     "feat_path": feat_path,
                     "wav_path": wav_path,
                     "vad": vad_label,
                     "turn": turn_label,
                     "last_ipu": last_ipu,
                     "target": turn_timing_target,
                     "das": das,
                     "da": da,
                     "nxt_das": nxt_das,
                     "nxt_da": nxt_da,
                    }

            batch_list.append(batch)
            
        return batch_list
    
    def get_data(self):
        data = []
        for file_name in tqdm(self.file_names):
            data += self.get_turn_info(file_name)
            
        return data            
        
    def __getitem__(self, index):
        batch = self.data[index]
        feat = np.load(batch['feat_path'])
        text = batch['text']
        idx = batch['idx']
        vad = batch['vad']
        turn = batch['turn']
        last_ipu = batch['last_ipu']
        target = batch['target']
        das = batch['das']
        da = batch['da']
        nxt_das = batch['nxt_das']
        nxt_da = batch['nxt_da']
        
        length = min(len(feat), len(vad), len(turn), len(target), len(text))
        batch['text'] = text[:length]
        batch['idx'] = idx[:length]
        batch['vad'] = vad[:length]
        batch['turn'] = turn[:length]
        batch['last_ipu'] = last_ipu[:length]
        batch['target'] = target[:length]
        batch['das'] = das[:length]
        batch['da'] = da[:length]
        batch['nxt_das'] = nxt_das[:length]
        batch['nxt_da'] = nxt_da[:length]
        batch['feat'] = feat[:length]
        batch['indices'] = index
        
        wav_len = int(length * self.sample_rate * self.frame_length / 1000)
        
        assert len(batch['feat'])==len(batch['vad']), "error"
        
        return list(batch.values())

    def __len__(self):
        return len(self.data)
    

def collate_fn(batch):
    chs, offsets, is_barge_in, texts, kanas, idxs, feat_paths, wav_paths, vad, turn, last_ipu, targets, das, da, nxt_das, nxt_da, feats, indices = zip(*batch)
    
    batch_size = len(chs)
    
    max_len = max([len(f) for f in feats])
    feat_dim = feats[0].shape[-1]
    da_dim = len(das[0])
    
    text_ = []
    # kana_ = []
    vad_ = torch.zeros(batch_size, max_len).long()
    turn_ = torch.zeros(batch_size, max_len).long()
    last_ipu_ = torch.zeros(batch_size, max_len).long()
    target_ = torch.ones(batch_size, max_len).long()*(-100)
    feat_ = torch.zeros(batch_size, max_len, feat_dim)
    das_ = torch.zeros(batch_size, max_len, da_dim)
    da_ = torch.zeros(batch_size, max_len, da_dim)
    nxt_das_ = torch.zeros(batch_size, max_len, da_dim)
    nxt_da_ = torch.zeros(batch_size, max_len, da_dim)
    
    input_lengths = []
    for i in range(batch_size):
        l1 = len(feats[i])
        input_lengths.append(l1)
                
        text_.append(texts[i]+['[PAD]']*(max_len-l1))
        vad_[i, :l1] = torch.tensor(vad[i]).long()       
        turn_[i, :l1] = torch.tensor(turn[i]).long()       
        last_ipu_[i, :l1] = torch.tensor(last_ipu[i]).long()
        target_[i, :l1] = torch.tensor(targets[i]).long()       
        feat_[i, :l1] = torch.tensor(feats[i])
        das_[i, :l1] = torch.tensor(das[i]).unsqueeze(0).repeat((l1, 1))
        da_[i, :l1] = torch.tensor(da[i]).unsqueeze(0).repeat((l1, 1))
        nxt_das_[i, :l1] = torch.tensor(nxt_das[i]).unsqueeze(0).repeat((l1, 1))
        nxt_da_[i, :l1] = torch.tensor(nxt_das[i]).unsqueeze(0).repeat((l1, 1))
            
    input_lengths = torch.tensor(input_lengths).long()
        
    return chs, text_, kanas, idxs, vad_, turn_, last_ipu_, target_, feat_, input_lengths, offsets, indices, is_barge_in, wav_paths, das_, da_, nxt_das_, nxt_da_
    

# torch.utils.data.DataLoader によってデータローダの作成
def create_dataloader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=2):
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn= lambda x: collate_fn(x),
    )
    return loader

# datasetの取得 -> クラスATRDatasetの作成
def get_dataset(config, cv_id, split='train', subsets=["M1_all"], speaker_list=None):
    dataset = ATRDataset(config, cv_id, split, subsets, speaker_list)
    return dataset

# dataloaderの取得 -> create_dataloaderによって作成
def get_dataloader(dataset, config, shuffle=True):
    dataloader = create_dataloader(dataset, config.optim_params.batch_size, shuffle=shuffle)
    return dataloader
