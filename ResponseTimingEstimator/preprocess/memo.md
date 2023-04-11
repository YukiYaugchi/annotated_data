### README

* make_annotated_data.py
  * ターン情報を取得
  
* make_annotated_turn_csv.py
  * make_annotated_data.pyで保存したターン情報から, データセットに使用するターン情報を切り取る
  
* make_annotated_turn_wav.py
  * make_annotated_turn_csv.pyで保存したターン情報を元にターン単位のwavを切り抜く
  * wavに対応する開始(wav_start), 終了時間(wav_end)を保存

* make_annotated_wav2text.py
  * espnetで学習したモデルでdecodeしてテキストを保存する
  
* make_annotated_cnnae.py
  * CNN-AEで特徴量を取得
  
* make_annotated_split_name.py
  * データ（name）をtrain(M1_train.txt), val(M1_val.txt), test(M1_test.txt)に分ける
  