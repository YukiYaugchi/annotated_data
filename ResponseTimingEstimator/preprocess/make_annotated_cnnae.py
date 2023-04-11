import wave
import numpy as np
import torch
import os
import glob
from tqdm import tqdm

#pectrogram の generator の定義
import sflib.sound.sigproc.spec_image as spec_image
generator = spec_image.SpectrogramImageGenerator()

# 学習済み AE の 読み込み
import sflib.speech.feature.autoencoder_pytorch.base as base

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
tr0006_18 = base.load(18, 'csj_0006', 'CSJ0006', device=device, map_location='cpu')
ae2 = tr0006_18.autoencoder

# wav folder の指定
DATAROOT = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator/data/ATR_Annotated/data_-500_2000'

names_dir = os.path.join(DATAROOT, 'wav')
names = os.listdir(names_dir)
for i, name in enumerate(tqdm(names)):
    turn_dir = os.path.join(names_dir, name)
    turns = os.listdir(turn_dir)
    for j, turn in enumerate(turns):
        wav_path = os.path.join(turn_dir, turn)
    
        output_dir = turn_dir.replace('wav', 'cnn_ae')
        os.makedirs(output_dir, exist_ok=True)
    
        spec_path =  os.path.join(output_dir, turn.replace('.wav', '_spec.npy'))
        pow_path =  os.path.join(output_dir, turn.replace('.wav', '_pow.npy'))

        if os.path.exists(spec_path):
            continue

        wf = wave.open(wav_path)
        x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
        pad = np.zeros(int(16000*0.05), np.int16)
        x = np.concatenate([pad, x, pad])
        
        with torch.no_grad():
            generator = spec_image.SpectrogramImageGenerator()
            #spectrogramの作成
            result = generator.input_wave(x)

            power = []
            feature = []
            #中間層出力 (encode)
            for j in range(len(result)):
                image_in = result[j].reshape(1, 512, 10)
                image_in = torch.tensor(image_in).float().to(device)
                # 中間層出力
                x, l2 = ae2.encode(image_in)
            
                power.append(l2[0].detach().cpu().data.numpy())
                feature.append(x[0].detach().cpu().data.numpy())

        power = np.vstack(power)
        feature = np.vstack(feature)
    
        #特徴量保存
        np.save(spec_path, feature)
        np.save(pow_path, power)
