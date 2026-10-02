# Planetary model checkpoints

Model binaries are distributed separately and are excluded from Git.

For automatic checkpoint selection, place the corresponding released files here
under these local names:

| Training domain | DenseDepth | DepthFormer | NeWCRFs |
| --- | --- | --- | --- |
| Moon | `moon-D.pth` | `moon-V.pth` | `moon-N.pth` |
| Mars | `mars-D.pth` | `mars-V.pth` | `mars-N.pth` |

Select a file with `--body moon|mars -m D|V|N`, or pass its actual location
with `--weights /path/to/checkpoint.pth`. The filenames above are a local
selection convention; naming a file does not establish its training domain
or change its architecture. Keep the published metadata for each checkpoint.

DenseDepth in this source tree uses DenseNet-161. Supply a matching checkpoint.
The DepthFormer branch requires a complete model checkpoint; encoder-only
`mit_b4.pth` weights are not a substitute for planetary model weights.

See the main [README](../README.md) for model-resource links and input preparation.
