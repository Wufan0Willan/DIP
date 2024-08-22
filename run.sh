fairseq-hydra-train \
    task.data=/home/wupeng/fairseq_DIP/dataset/PT_vox2k_full \
    --config-dir /home/wupeng/fairseq/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_loss10_neg100f.yaml \
