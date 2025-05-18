### To run the file
```
python main.py data.path=data/processed.zarr
```

### Validating after training
```
python main.py \
  data.path=data/processed.zarr \
  ckpt_path=outputs/2025-05-16/21-57-50/checkpoints/last.ckpt \
  trainer.max_epochs=0

```