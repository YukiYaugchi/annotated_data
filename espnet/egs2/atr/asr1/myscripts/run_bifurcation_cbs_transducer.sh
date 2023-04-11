#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#stage=12
skip_data_prep=false
#train_set=train_sp
train_set=train
valid_set=valid
test_sets=test


asr_config=myconf/train_asr_bifurcation_cbs_transducer.yaml
inference_config=myconf/decode_rnnt_conformer_streaming.yaml
inference_asr_model=valid.loss_transducer.ave_10best.pth

bpe_train_text=dump/raw/train_sp/text
# bpe_train_text=dump/raw/train/text
lm_config=myconf/train_lm.yaml
use_lm=false
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh                                               \
    --ngpu 3                                           \
    --stage 2                                          \
    --stop_stage 4                                    \
    --skip_data_prep "${skip_data_prep}"               \
    --my_token_listdir "my_token_list"                 \
    --use_transducer true                              \
    --use_parallel_transducer false                    \
    --use_dual_delay false                             \
    --use_streaming false                              \
    --lang jp                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --use_eou false                                    \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --inference_asr_model "${inference_asr_model}"     \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/${train_set}/text" "$@"      \
    #--pretrained_model "${init_model}"                 \
    #--asr_speech_fold_length 512 \
    #--asr_text_fold_length 150 \
    #--lm_fold_length 150 \
    #--lm_train_text "data/${train_set}/text" "$@"
    #--nbpe 300                                         \
    #--bpe_train_text ${bpe_train_text}        \
