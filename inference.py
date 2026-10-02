#!/usr/bin/env python3
"""MISTE relative-height inference for 640 x 480 planetary image tiles."""

import argparse
from pathlib import Path

from miste_checkpoints import add_checkpoint_arguments, resolve_checkpoint


def normalise(value):
    """Normalize one predicted height tile to [0, 1]."""
    value = value.cpu().numpy()[0, :, :]
    vmin, vmax = value.min(), value.max()
    return (value - vmin) / (vmax - vmin) if vmin != vmax else value * 0


def image_loader(image_name):
    """Read a fixed-size input tile and replicate a grayscale band as RGB."""
    from PIL import Image
    from torchvision import transforms

    with Image.open(image_name) as source:
        if source.size != (640, 480):
            raise ValueError(f'{image_name}: expected 640 x 480, received {source.size}')
        image = source.convert('RGB')
        return transforms.ToTensor()(image).float().unsqueeze(0)


def predict_image(model, image, args):
    import torch
    import torch.nn.functional as functional

    with torch.no_grad():
        image = image.to(args.device)
        prediction = model(image)
        if args.model == 'V':
            _, prediction, _ = prediction
        return functional.interpolate(prediction.cpu(), image.shape[-2:],
                                      mode='bilinear', align_corners=True)


def save_output_image(prediction, out_file):
    import numpy as np
    from PIL import Image

    Image.fromarray(prediction.astype(np.float32)).save(out_file)


def load_model(args):
    """Load the original architecture and its complete trained state dictionary."""
    import torch

    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    if args.model == 'D':
        from DenseNet161UNet.model import DenseDepth
        model = DenseDepth(encoder_pretrained=False)
        state = checkpoint.get('model_state_dict', checkpoint)
    elif args.model == 'V':
        from ViTABUNet.model import UnetAdaptiveBins
        model = UnetAdaptiveBins.build(n_bins=256, min_val=1, max_val=254,
                                      norm='linear', encoder_weights=None)
        state = checkpoint.get('model', checkpoint)
    else:
        from NWFCCRF.model import NewCRFDepth
        model = NewCRFDepth(version='large07', inv_depth=False, max_depth=255)
        state = checkpoint.get('model', checkpoint)

    state = {key.removeprefix('module.'): value for key, value in state.items()}
    if args.model == 'V':
        state = {
            key.replace('adaptive_bins_layer.embedding_conv.', 'adaptive_bins_layer.conv3x3.')
               .replace('adaptive_bins_layer.patch_transformer.embedding_encoder',
                        'adaptive_bins_layer.patch_transformer.embedding_convPxP'): value
            for key, value in state.items()
        }
    model.load_state_dict(state, strict=True)
    return model.to(args.device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_checkpoint_arguments(parser)
    parser.add_argument('--device', '-d', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--data', default='test_data/', help='Directory of input TIFF tiles')
    args = parser.parse_args()
    args.weights = resolve_checkpoint(args, parser)
    data = Path(args.data).expanduser().resolve()
    if not data.is_dir():
        parser.error(f'Input tile directory not found: {data}')
    images = sorted(path for path in data.iterdir()
                    if path.suffix.lower() in {'.tif', '.tiff'}
                    and not path.stem.endswith('_result'))
    if not images:
        parser.error(f'No input TIFF tiles found in {data}')

    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use --device cpu or a CUDA-enabled environment.')
    model = load_model(args)
    print(f'MISTE: model={args.model}, device={args.device}, checkpoint={args.weights}')
    for index, path in enumerate(images, 1):
        prediction = predict_image(model, image_loader(path), args)
        out = path.with_name(path.stem + '_result' + path.suffix)
        save_output_image(normalise(prediction).squeeze(), out)
        print(f'[{index}/{len(images)}] {out}')


if __name__ == '__main__':
    main()
