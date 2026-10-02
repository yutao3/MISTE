"""Checkpoint selection for MISTE planetary terrain estimation."""

from pathlib import Path


def add_checkpoint_arguments(parser):
    """Register the model and planetary checkpoint arguments."""
    parser.add_argument('-m', '--model', choices=['D', 'V', 'N'], default='D',
                        help='D: DenseDepth; V: DepthFormer; N: NeWCRFs (default: D)')
    parser.add_argument('-w', '--weights',
                        help='Explicit checkpoint path; overrides the body-based convention')
    parser.add_argument('--body', choices=['moon', 'mars'],
                        help='Training domain for automatic checkpoint selection')
    parser.add_argument('--weights-dir', default=str(Path(__file__).resolve().parent / 'pre-trained-weights'),
                        help='Directory containing <body>-<model>.pth checkpoints')


def resolve_checkpoint(args, parser):
    """Resolve an explicit path or a body/model pair and require an existing file."""
    if args.weights:
        checkpoint = Path(args.weights).expanduser()
    elif args.body:
        checkpoint = Path(args.weights_dir).expanduser() / f'{args.body}-{args.model}.pth'
    else:
        parser.error('Specify --weights PATH or --body moon|mars.')
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f'Checkpoint not found: {checkpoint}. Download the trained weights separately.')
    return str(checkpoint)
