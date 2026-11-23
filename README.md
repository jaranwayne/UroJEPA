# UroJEPA

Code for **"A self-supervised three-dimensional computed tomography foundation model for predicting urosepsis, difficult ureteral access, and spontaneous stone passage"**.

UroJEPA is a V-JEPA 2.1 ViT-B/16 encoder pre-trained on unlabelled non-contrast CT videos and fine-tuned for four preoperative outcomes in ureteral stone disease: stone-associated sepsis, urosepsis after ureteroscopic lithotripsy, difficult ureter, and failed spontaneous passage.

* **UroJEPA**: CT pre-training from random weights
* **UroJEPA†**: CT pre-training from the public natural-video V-JEPA 2.1 weights

Built on [V-JEPA 2 / 2.1](https://github.com/facebookresearch/vjepa2). No patient data or trained weights are included.

## Installation

```bash
git clone https://github.com/jaranwayne/UroJEPA.git && cd UroJEPA
pip install -r requirements.txt      # install PyTorch for your CUDA version first
python -m pytest                     # optional checks
```

## Data preparation

**1. DICOM to video.** Each CT series (axial or coronal, 5-mm slices) becomes one MP4 whose frames are the slices.

```bash
# one series
python scripts/dcm2video.py --dicom-dir /dicom/P0001/axial --out data/videos/P0001/axial.mp4
# all series under a folder (/dicom/<patient>/<series>/*.dcm -> data/videos/<patient>/<series>.mp4)
python scripts/dcm2video.py --dicom-root /dicom --out-root data/videos --workers 8
```

The default parameters (window 50/180, 512 × 512, 10 fps) are those used in the paper. Per-scanner settings go in `configs/dcm2video.yaml`.

**2. Manifests and splits.**

```bash
# unlabelled pre-training videos (exclude patients of the outcome cohorts)
python scripts/build_manifest.py --video-dir data/videos --output data/pretrain_manifest.csv \
    --exclude-patients data/labelled_patients.csv --min-frames 16

# one labelled CSV per outcome: patient_id,video_path,label  ->  7:1:2 stratified split
python scripts/build_manifest.py --input data/stone_sepsis_labels.csv --output data/stone_sepsis_labels.csv
python scripts/make_splits.py --labels data/stone_sepsis_labels.csv --out data/splits/stone_sepsis
```

Put an external cohort, if any, at `data/splits/<task>/external.csv`. Tasks: `stone_sepsis`, `urosepsis_ursl`, `difficult_ureter`, `spontaneous_passage`.

## Pre-training

```bash
# UroJEPA
torchrun --standalone --nproc_per_node=8 pretrain.py --config configs/pretrain/urojepa_vitb16.yaml

# UroJEPA†
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt -P checkpoints/
torchrun --standalone --nproc_per_node=8 pretrain.py --config configs/pretrain/urojepa_dagger_vitb16.yaml
```

Clips are 16 slices at two scales (5-mm and 10-mm spacing), sampled patient → video → scale → clip.

## Fine-tuning and evaluation

```bash
# one outcome, one seed
python finetune.py --config configs/finetune/tasks/stone_sepsis.yaml \
    --opts seed=0 model.pretrained=outputs/pretrain/urojepa_dagger_vitb16/latest.pth.tar \
           folder=outputs/finetune/stone_sepsis/urojepa_dagger

# four outcomes x five seeds
bash scripts/run_finetune.sh urojepa_dagger outputs/pretrain/urojepa_dagger_vitb16/latest.pth.tar

# label efficiency (10/25/50/75 % of training patients)
bash scripts/run_label_efficiency.sh urojepa_dagger outputs/pretrain/urojepa_dagger_vitb16/latest.pth.tar

# mean ± SD over seeds
python scripts/summarize.py --root outputs/finetune

# apply a fine-tuned model to new data
python evaluate.py --checkpoint outputs/finetune/stone_sepsis/urojepa_dagger/seed0/best.pt \
    --csv external=data/new_center.csv --out outputs/eval/stone_sepsis
```

The encoder and an attentive probe are fine-tuned end to end. The checkpoint with the best validation AUROC is evaluated on the test split and external cohorts. Logits are averaged over 9 clips per video. Each run writes `metrics.json` and `predictions_<cohort>.csv`. Any config value can be changed with `--opts key=value`.

## Citation

```bibtex
@article{lin2026urojepa,
  title  = {A Self-supervised Three-dimensional Computed Tomography Foundation Model for Predicting
            Urosepsis, Difficult Ureteral Access, and Spontaneous Stone Passage: A Multicenter Study},
  author = {Lin, Yunqiao and Wang, Zheyuan and Chen, Jianhui and others},
  year   = {2026}
}
```

Please also cite V-JEPA 2.1 ([arXiv:2603.14482](https://arxiv.org/abs/2603.14482)) and V-JEPA 2 ([arXiv:2506.09985](https://arxiv.org/abs/2506.09985)).

## License

MIT (see `LICENSE`). Code adapted from V-JEPA 2 / 2.1 keeps its original copyright notice. For research use only; not a medical device.
