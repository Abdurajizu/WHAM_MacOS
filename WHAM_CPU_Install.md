# WHAM on macOS (CPU) Setup

## 1) Create environment
```bash
cd /Users/abdulazizsobirov/WHAM-main
bash install_macos_cpu.sh
conda activate wham-mac
```

## 2) Download models/assets
```bash
bash fetch_demo_data.sh
```

If SMPL auth download fails, manually place these files in `dataset/body_models/smpl/`:
- `SMPL_NEUTRAL.pkl`
- `SMPL_FEMALE.pkl`
- `SMPL_MALE.pkl`

## 3) Run inference (CPU, Mac-safe)
```bash
python demo.py \
  --video /absolute/path/to/input.mp4 \
  --output_pth /Users/abdulazizsobirov/WHAM-main/output/demo \
  --device cpu \
  --detector_backend yolo_pose \
  --estimate_local_only \
  --save_pkl
```

## 4) Render output video (optional, needs PyTorch3D)
```bash
python demo.py \
  --video /absolute/path/to/input.mp4 \
  --output_pth /Users/abdulazizsobirov/WHAM-main/output/demo \
  --device cpu \
  --detector_backend yolo_pose \
  --estimate_local_only \
  --save_pkl \
  --visualize
```

Output path:
- `/Users/abdulazizsobirov/WHAM-main/output/demo/<video_name>/output.mp4`
