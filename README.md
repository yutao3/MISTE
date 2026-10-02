# MISTE

**A Pre-Trained Deep Learning Toolbox for Planetary Monocular Image to Surface Topography Estimation**

MISTE is a research toolbox for generating or refining planetary digital terrain models (DTMs/DEMs) from a single map-projected orbital image and a corresponding lower-resolution reference DTM. The reference provides absolute elevation and broad terrain structure; pre-trained monocular depth estimation models supply image-guided surface detail. The processing framework includes overlapping image tiles, optional inpainting, relative-height inference, georeferencing, reference-constrained postprocessing, mosaicing, and coarse-to-fine multi-scale processing.

MISTE extends the [LISTER toolbox](https://github.com/yutao3/LISTER) to support separate Mars-trained and Moon-trained checkpoints. The accompanying study evaluates Mars and lunar imagery and explores transfer of Moon-trained models to Vesta, Europa, and Enceladus. A checkpoint's training domain is selected independently of the body's map projection: all inputs must retain the correct planetary coordinate reference system.

## Source distribution

This repository contains three inference branches and the processing utilities inherited from LISTER:

| Model ID | Model | Source directory |
| --- | --- | --- |
| `D` | DenseDepth, the default | `DenseNet161UNet/` |
| `V` | DepthFormer / ViT with adaptive depth bins | `ViTABUNet/` |
| `N` | NeWCRFs / windowed fully connected CRFs | `NWFCCRF/` |

The paper also evaluates **Marigold** and **DiffusionE2EFT**. Their planetary implementations and checkpoints are distributed separately; the `inference.py` supplied here accepts `D`, `V`, and `N` only. Selecting a diffusion model requires its corresponding inference implementation and environment.

The main `MISTE_autoDTM.py` pipeline retains the Gaussian low-/high-frequency merge from the public source. The `postprocess_poly*.py` files provide standalone polynomial-fitting alternatives; they are not invoked automatically by the main pipeline. In particular, the existing automatic polynomial helper selects the lower-RMSE fit without the additional improvement threshold described in the paper. Consequently, this source distribution alone does not reproduce every experiment in the manuscript.

The supplied DenseDepth branch uses a **DenseNet-161** encoder, as implemented in the source, whereas the manuscript describes DenseNet-169. The architecture is retained to preserve compatibility with existing checkpoints. A DenseNet-169 checkpoint requires its matching model implementation.

## Installation

Use a Linux environment with Python 3.9 or later. The core pipeline requires:

- GDAL command-line tools and Python bindings.
- [NASA Ames Stereo Pipeline (ASP)](https://github.com/NeoGeographyToolkit/StereoPipeline), including `dem_mosaic`.
- PyTorch and a compatible torchvision installation. CUDA is the default inference device; CPU inference can be requested with `--device cpu`.
- The Python packages in `requirements.txt`.

[USGS ISIS](https://github.com/DOI-USGS/ISIS3) is required only for the raw LROC NAC preparation utility. It is not needed when the input is already a prepared GeoTIFF.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the GDAL Python bindings against the system GDAL version. The machine-learning and geospatial packages must be compatible with the local Python, CUDA, and GDAL installation; `requirements.txt` is not a frozen reproduction environment.

The `N` branch additionally imports `mmcv.cnn.ConvModule`. Install a compatible MMCV environment following the [NeWCRFs installation instructions](https://github.com/aliyun/NeWCRFs#installation). MMCV is loaded only for `N` inference, so it is not required for the `D` or `V` branches.

Check the command-line interfaces with:

```bash
python MISTE_autoDTM.py --help
python MISTE_autoDTM_MSP.py --help
python inference.py --help
```

## Pre-trained weights and data

Weights and large datasets are stored separately from the Git source tree. The original distribution links to the following resources:

- [Model resources linked by LISTER](https://drive.google.com/drive/folders/1uQZtQKEiKxk3WJoZn2wMLVHRkegJQU6G?usp=drive_link).
- [Mars and Moon training data linked in the MISTE manuscript](https://box.fu-berlin.de/s/4wCAmoQ8BP38aRC).

The training-data link is not an automatic checkpoint downloader. Use the appropriate released planetary checkpoint for the selected architecture. This repository does not include weights or create them from a body name.

There are two ways to supply a checkpoint:

1. Pass its actual path with `-w /path/to/checkpoint.pth`. The filename is unrestricted.
2. Store it under the local naming convention below and select `--body mars` or `--body moon`.

| Training domain | DenseDepth (`D`) | DepthFormer (`V`) | NeWCRFs (`N`) |
| --- | --- | --- | --- |
| Moon | `moon-D.pth` | `moon-V.pth` | `moon-N.pth` |
| Mars | `mars-D.pth` | `mars-V.pth` | `mars-N.pth` |

By default these names are resolved beneath `pre-trained-weights/` next to the scripts. Use `--weights-dir DIRECTORY` for a different location. These are local naming conventions, not a listing of verified downloadable checkpoint filenames. An explicit `--weights` path takes precedence over `--body`.

`--body` identifies the checkpoint's **training domain**. It does not reproject input data, change a planetary radius, or train a new model. To run a Moon-trained model on another body, use `--body moon` and prepare the image/reference DTM in that body's correct map projection.

The inference loader accepts the original DenseDepth `model_state_dict`, the DepthFormer/NeWCRFs `model` state dictionary, or a bare state dictionary. It removes a leading `module.` prefix and retains the original DepthFormer naming compatibility rules. Complete model weights are loaded strictly; missing or incompatible architecture parameters produce an error. DepthFormer inference loads its encoder from the complete model checkpoint and does not additionally require `mit_b4.pth`.

## Input preparation

The primary pipelines expect:

- An **8-bit, single-band, map-projected GeoTIFF**, at least 640 pixels wide and 480 pixels high.
- A co-registered, single-band reference DTM in the **same planetary CRS**, with compatible elevation units and vertical reference.
- A north-up raster grid with no rotated geotransform. Prepare/reproject other grids before running the pipeline.
- A reference DTM covering the entire region being processed.

The inherited image-mask convention treats **zero-valued image pixels as NoData**. Scale valid grayscale values to 1-255 and reserve 0 for gaps; choose a stretch that preserves useful terrain texture. Declaring a different NoData tag alone does not change this image-mask convention.

The model consumes 640 x 480 tiles. Predictions are normalized relative-height estimates; they have no independent absolute elevation scale until constrained by the reference DTM. The final elevation values inherit the reference's height units and datum.

For raw lunar LROC NAC EDR images, a preparation utility and lunar ISIS map templates are supplied:

```bash
python prepare_geotiff_lroc_nac.py input.IMG lroc_southpole.map
```

Use the appropriate lunar projection template for the area. These templates and LROC download utilities are specific to the Moon; Mars and other planetary data must be prepared with their own sensor calibration and projection workflows.

## Single-scale processing (SSP)

Use SSP when the reference DTM is reasonably close to the input image resolution. The study discusses reference grids approximately 2-5 times coarser than the image as a typical refinement case.

```bash
python MISTE_autoDTM.py \
  -i mars_image_8bit.tif -r mars_reference_dtm.tif \
  -o output/mars_miste_dtm.tif \
  --body mars -a /path/to/ASP/bin --inpaint
```

To use an explicit lunar checkpoint:

```bash
python MISTE_autoDTM.py \
  -i lunar_image_8bit.tif -r lunar_reference_dtm.tif \
  -o output/lunar_miste_dtm.tif \
  -m D -w /path/to/lunar_checkpoint.pth \
  -a /path/to/ASP/bin --inpaint
```

The main pipeline performs tiling, relative-height inference, georeferencing, Gaussian reference/detail merging, and ASP mosaicing. It creates an isolated work directory beneath `--tmp` and removes that work directory on success. Failed single-scale runs retain their intermediate files for inspection.

## Multi-scale processing (MSP)

Use MSP when the reference DTM is substantially coarser than the input image or broad terrain structure needs to be reconstructed before local refinement. Each finer scale uses the DTM from the preceding scale as its reference.

```bash
python MISTE_autoDTM_MSP.py \
  -i mars_image_8bit.tif -r coarse_mars_reference.tif \
  -o output/mars_miste_msp.tif \
  --body mars -a /path/to/ASP/bin \
  --num_of_scales 3 --inpaint
```

For Moon-trained checkpoints, replace `--body mars` with `--body moon`. For a particular checkpoint at any location, use `-w PATH` instead.

The MSP wrapper derives a starting scale from image shape and reference pixel spacing, then processes toward full image resolution. The inherited reference constraint aims for a coarsest image pixel spacing around one third of the reference spacing or finer. The downsampling ratio is capped at 1. Images with width or height below 1000 pixels use a single-scale call. This still requires an image of at least 640 x 480 pixels.

Intermediate DTM outputs use `<output_base>_level0.tif`, `<output_base>_level1.tif`, and so on. The final level is written to `--output`. The main MSP wrapper removes its isolated temporary work directory on completion or failure; intermediate DTM outputs beside the requested output remain available.

### Primary pipeline options

| Option | Default | Purpose |
| --- | --- | --- |
| `-i`, `--input` | Required | Prepared image GeoTIFF |
| `-r`, `--ref` | Required | Reference DTM |
| `-o`, `--output` | Required | Final DTM output path |
| `-m`, `--model` | `D` | Model branch: `D`, `V`, or `N` |
| `-w`, `--weights` | Unset | Explicit model checkpoint |
| `--body` | Unset | Automatic selection: `moon` or `mars` |
| `--weights-dir` | Script-relative `pre-trained-weights/` | Checkpoint directory |
| `--device` | `cuda` | Inference device: `cuda` or `cpu` |
| `-a`, `--asp` | `~/Downloads/ASP/bin` | Directory containing `dem_mosaic` |
| `-t`, `--tmp` | `data_tmp` | Parent for temporary work directories |
| `--overlap` | `280` | Tile overlap; valid range 0-479 pixels |
| `--valid_threshold` | `20` | Intensity threshold for the inpainting eligibility count |
| `--max_nodata_pixels` | `3000` | Reject tiles with at least this many remaining zero pixels |
| `--ndv` | Approximately `-3.4e+38` | NoData for relative-height tiles |
| `--scale` | `3.75` | Gaussian scale factor; multiplied by remaining levels in MSP |
| `--inpaint` | Disabled | Fill eligible image gaps temporarily before inference |
| `--inpaint_threshold` | `0.1` | Minimum valid fraction for inpainting |
| `--inpaint_method` | `telea` | `telea` or `ns` |
| `--fill_smoothing` | `1` | Reference FillNodata smoothing iterations |
| `--num_of_scales` | `3` | MSP only: number of pyramid levels |

For a negative scientific-notation NoData value, use `--ndv=-3.4e+38`. The `--scale` option in these pipelines controls Gaussian merging; the similarly named positional argument in the standalone polynomial scripts scales the relative-height amplitude.

## Standalone inference and utilities

Run inference on prepared 640 x 480 TIFF tiles:

```bash
python inference.py --data tiles/ --body mars -m D --device cuda
```

Relative predictions are saved beside input tiles as `<tile>_result.tif`. Existing result files are excluded from the input list.

The following examples use the actual script paths in this repository:

```bash
# Shadow-aware brightness adjustment
python pre_adjust_brightness.py input.tif adjusted.tif -m mask_gamma \
  --shadow-quantile 0.25 --shadow-high-quantile 0.45 --gamma 0.45 --sigma 20

# Image quality screening
python check_low_quality_robust.py input_images/ quality_report.txt

# First-/second-order polynomial alignment of a relative-height GeoTIFF
python postprocess_poly_auto.py reference.tif relative_georeferenced.tif fused.tif 5.0 True

# DTM comparisons over different smoothing widths
python additional_functions/compare2dtm.py reference.tif target.tif 15

# Visual and statistical comparison over an overlapping footprint
python validation/quick_validation.py validation_output/ input_8bit.tif reference.tif dtm1.tif dtm2.tif

# Image-level SSIM and RMSE
python additional_functions/cal_stat.py image_a.tif image_b.tif

# Refresh the lunar LROC NAC URL catalogue
python get_latest_lroc_nac_list_and_merge.py nac_catalogue/

# Download selected LROC NAC observations
python get_lroc_nac.py nac_catalogue/all_lroc_nac_urls.txt wanted_ids.txt nac_downloads/
```

The polynomial utilities retain their original experimental behaviour. `postprocess_poly_auto_scale.py` additionally rescales the final height range to the reference range with a fixed margin; this differs from `postprocess_poly_auto.py`. Both require appropriate preparation of valid input/reference pixels. The image-level statistics utility reads grayscale images and is not a geospatial elevation-accuracy validator.

## Repository contents

| Path | Role |
| --- | --- |
| `MISTE_autoDTM.py` | Primary single-scale pipeline |
| `MISTE_autoDTM_MSP.py` | Primary multi-scale wrapper |
| `MISTE_autoDTM_MSP_light.py` | Earlier simplified MSP variant |
| `MISTE_autoDTM_original_smoothing_merge.py` | Earlier smoothing/merge variant |
| `inference.py` | Shared `D`/`V`/`N` relative-height inference |
| `miste_checkpoints.py` | Checkpoint argument and path selection |
| `DenseNet161UNet/`, `ViTABUNet/`, `NWFCCRF/` | Model architectures |
| `full_chain.py`, `make_tiles.py`, `georeference.py`, `postprocess*.py` | Original modular workflow and postprocessing variants |
| `prepare_geotiff_lroc_nac.py`, `lroc_*.map` | Lunar raw-image preparation |
| `get_lroc_nac.py`, `get_latest_lroc_nac_list_and_merge.py` | Lunar image discovery/download |
| `pre_adjust_brightness.py`, `check_low_quality_robust.py` | Image preprocessing and screening |
| `additional_functions/`, `validation/` | Comparison and diagnostic utilities |

The two primary pipelines are the recommended command interfaces. Earlier variants remain available for continuity and have their own options and temporary-file handling; the primary pipeline guarantees described above do not apply to all legacy scripts.

## Interpretation and validation

MISTE outputs are **AI-enhanced, reference-constrained topographic estimates**. Fine detail may reflect surface relief, albedo, shadows, sensor artefacts, or domain shift. The study's cross-body demonstrations are exploratory and do not establish equivalent quantitative accuracy on every planetary surface.

A final grid matching the input image spacing does not by itself prove equivalent effective spatial resolution or height accuracy. Validate scientific measurements against independent stereo DTMs, laser-altimeter observations, or rover/lander products where available. A comparison with the reference used in reconstruction measures consistency, not fully independent validation.

## Citation

If you use MISTE in research, please cite the accompanying manuscript:

> Tao, Y., Walter, S. H. G., Muller, J.-P., and Xiong, S. (2026). *A Pre-Trained Deep Learning Toolbox for Planetary Monocular Image to Surface Topography Estimation (MISTE).* Manuscript.

Citation metadata is provided in `CITATION.cff`. Also cite the model and processing tools used in your workflow.

## Acknowledgements

Mars surface-topography development and processing were supported by the German Space Agency (DLR Bonn), Grant **50OO2204 (Koregistrierung)**, on behalf of the German Federal Ministry for Economic Affairs and Climate Action (2023-2026). Lunar surface-topography development and processing were supported by the European Space Agency's **LISTER** project, Contract **4000145348**, at Surrey AI Imaging Ltd (2024). Reprocessing of the lunar south-polar Shackleton region was partially supported by NASA/JPL, Contract **1697157**, at UCL and SAIIL (2023). Initial development of the Mars surface-topography MADNet network was funded by the UK Space Agency Aurora Programme, Grant **ST/S001891/1**, at UCL (2018-2021).

Part of the work was carried out at the Guangdong Laboratory of Artificial Intelligence and Digital Economy, China, with support from the Guangdong Basic and Applied Basic Research Foundation, Grant **2024A1515011679**. Acquisition of the CE-2 and LROC NAC datasets was supported by Dr Kaichang Di and Dr Bin Liu of the State Key Laboratory of Remote Sensing Science, Chinese Academy of Sciences.

We acknowledge the authors and maintainers of [DenseDepth](https://github.com/ialhashim/DenseDepth), [DepthFormer](https://github.com/ashutosh1807/Depthformer), [NeWCRFs](https://github.com/aliyun/NeWCRFs), [Marigold](https://github.com/prs-eth/Marigold), [DiffusionE2EFT](https://github.com/VisualComputingInstitute/diffusion-e2e-ft), and [NASA Ames Stereo Pipeline](https://github.com/NeoGeographyToolkit/StereoPipeline).

## Licence and provenance

This distribution retains the licensing terms and licence texts supplied with LISTER:

- [Apache License 2.0](LICENSE.Apache-2.0).
- [ESA Software Community Licence - Weak Copyleft](LICENSE.ESA-Community-Weak-Copyleft).

The original repository offers a choice of these licences for its software. Third-party components and externally distributed model resources retain their respective licences. See `NOTICE` for source provenance.
