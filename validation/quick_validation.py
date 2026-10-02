import os
import sys
import random
import rasterio
import numpy as np
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.transform import Affine
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.metrics import structural_similarity as ssim
from skimage import exposure
from scipy.ndimage import gaussian_filter, generic_filter

def print_usage_and_exit():
    print('Usage:')
    print('  python quick_validation.py output_dir input_image.tif input_DTM_1.tif input_DTM_2.tif ...')
    sys.exit(1)

def load_raster(path, expected_dtype=None, expected_bands=1):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        if expected_dtype and arr.dtype != expected_dtype:
            raise ValueError(f'{os.path.basename(path)}: Unexpected dtype {arr.dtype}, expected {expected_dtype}')
        if ds.count != expected_bands:
            raise ValueError(f'{os.path.basename(path)}: Expected {expected_bands} band(s), found {ds.count}')
        nodata = ds.nodata
        return arr, ds.transform, ds.crs, ds.res, ds.width, ds.height, ds.bounds, nodata

def get_overlap_bounds(bounds_list):
    min_left = max([b.left for b in bounds_list])
    max_right = min([b.right for b in bounds_list])
    min_bottom = max([b.bottom for b in bounds_list])
    max_top = min([b.top for b in bounds_list])
    if min_left >= max_right or min_bottom >= max_top:
        raise ValueError('No overlapping area found between all inputs.')
    return rasterio.coords.BoundingBox(min_left, min_bottom, max_right, max_top)

def get_highest_resolution(res_list):
    res_x = min([r[0] for r in res_list])
    res_y = min([r[1] for r in res_list])
    return (res_x, res_y)

def main():
    print("Starting quick_validation.py ...")
    if len(sys.argv) < 5:
        print_usage_and_exit()
    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)
    input_image_path = sys.argv[2]
    dtm_paths = sys.argv[3:]
    if len(dtm_paths) < 2:
        print('Please provide at least 2 DTM files (1 reference + 1 to compare).')
        sys.exit(1)
    reference_dtm_path = dtm_paths[0]
    compare_dtm_paths = dtm_paths[1:]

    print(f"Output directory: {output_dir}")
    print(f"Reference image: {input_image_path}")
    print(f"Reference DTM: {reference_dtm_path}")
    print(f"Comparison DTMs: {', '.join(compare_dtm_paths)}")

    # --- Read headers and basic info ---
    print("Loading and checking inputs ...")
    image_arr, image_transform, image_crs, image_res, image_w, image_h, image_bounds, image_nodata = load_raster(input_image_path, np.uint8, 1)
    ref_arr, ref_transform, ref_crs, ref_res, ref_w, ref_h, ref_bounds, ref_nodata = load_raster(reference_dtm_path, np.float32, 1)
    dtm_infos = []
    for p in compare_dtm_paths:
        arr, transform, crs, res, w, h, bounds, nodata = load_raster(p, np.float32, 1)
        dtm_infos.append((arr, transform, crs, res, w, h, bounds, nodata))

    # --- Get overlap bounds ---
    print("Calculating overlap region and target resolution ...")
    all_bounds = [image_bounds, ref_bounds] + [info[6] for info in dtm_infos]
    overlap_bounds = get_overlap_bounds(all_bounds)

    # --- Get highest resolution ---
    all_res = [image_res, ref_res] + [info[3] for info in dtm_infos]
    target_res = get_highest_resolution(all_res)
    print(f"Overlap bounds: {overlap_bounds}")
    print(f"Target resolution (m): {target_res}")

    # --- Prepare a common grid (target CRS, shape, transform) ---
    target_crs = ref_crs
    minx, miny, maxx, maxy = overlap_bounds
    width = int(np.ceil((maxx - minx) / target_res[0]))
    height = int(np.ceil((maxy - miny) / target_res[1]))
    target_transform = Affine(target_res[0], 0, minx, 0, -target_res[1], maxy)
    target_shape = (height, width)

    # --- If large dataset, process on a random 10-20% subwindow ---
    MAX_PIXELS = 3000 * 3000  # ~9 million pixels
    subregion_used = False
    if width * height > MAX_PIXELS:
        frac = random.uniform(0.1, 0.2)
        subw = int(width * np.sqrt(frac))
        subh = int(height * np.sqrt(frac))
        start_x = random.randint(0, width - subw)
        start_y = random.randint(0, height - subh)
        subwindow = (start_y, start_x, subh, subw)
        print(f"Large dataset detected. Using subregion: row {start_y}:{start_y+subh}, col {start_x}:{start_x+subw}")
        print(f"Subregion is ~{100*frac:.1f}% of the overlap area.")
        # Update bounds/shape/transform for subregion
        minx_new = minx + start_x * target_res[0]
        maxy_new = maxy - start_y * target_res[1]
        width = subw
        height = subh
        minx, maxy = minx_new, maxy_new
        target_transform = Affine(target_res[0], 0, minx, 0, -target_res[1], maxy)
        target_shape = (height, width)
        subregion_used = True
    else:
        print(f"Processing full overlap area ({width}x{height} pixels).")

    # --- Read and resample all inputs onto this grid ---
    print("Resampling and cropping all rasters to common grid ...")
    def warp_raster(path):
        with rasterio.open(path) as src:
            arr = src.read(
                1,
                out_shape=target_shape,
                resampling=Resampling.bilinear,
                window=rasterio.windows.from_bounds(
                    minx, miny, minx + width*target_res[0], maxy - height*target_res[1], transform=src.transform
                )
            )
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
        return arr

    image_c = warp_raster(input_image_path)
    ref_dtm_c = warp_raster(reference_dtm_path)
    dtm_c_list = [warp_raster(p) for p in compare_dtm_paths]

    print("All inputs successfully aligned and resampled.")

    # --- Preprocessing: mask out NaNs/NoData/Inf if present ---
    mask = np.isfinite(ref_dtm_c)
    for arr in dtm_c_list:
        mask &= np.isfinite(arr)
    mask &= np.isfinite(image_c)
    if image_nodata is not None:
        mask &= (image_c != image_nodata)
    mask &= (image_c > 0)  # assume image background is 0
    for arr in dtm_c_list:
        arr[~mask] = np.nan
    ref_dtm_c[~mask] = np.nan
    image_c[~mask] = 0

    # --- STATISTICS ---
    print("Calculating statistics for all DTMs ...")

    def calc_slope(dtm, res):
        # Only compute slope on valid pixels
        dtm = np.where(np.isfinite(dtm), dtm, np.nan)
        dzdx = np.gradient(dtm, axis=1) / res[0]
        dzdy = np.gradient(dtm, axis=0) / res[1]
        slope = np.arctan(np.sqrt(np.square(dzdx) + np.square(dzdy))) * 180 / np.pi
        slope[~np.isfinite(dtm)] = np.nan
        return slope

    def local_std(arr, win=5):
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return generic_filter(arr, np.nanstd, size=win, mode='nearest')

    def normalise(x):
        # Avoid nan/inf and constant images
        x = np.where(np.isfinite(x), x, 0)
        x = x - np.nanmin(x)
        rng = np.nanmax(x)
        return x / rng if rng != 0 else x

    smoothed_image = gaussian_filter(image_c.astype(float), sigma=2)

    stat_names = [
        'RMSE', 'MAE', 'SSIM_with_ref_DTM', 
        'Mean_Slope', 'Std_Slope', 'Mean_Roughness', 'Std_Roughness', 
        'SSIM_with_smoothed_image'
    ]
    stats_results = []

    # Calculate for reference DTM
    ref_slope = calc_slope(ref_dtm_c, target_res)
    ref_rough = local_std(ref_dtm_c, 5)
    ref_stats = [np.nan, np.nan, np.nan, np.nanmean(ref_slope[mask]), np.nanstd(ref_slope[mask]), np.nanmean(ref_rough[mask]), np.nanstd(ref_rough[mask]), np.nan]
    stats_results.append(['REF_DTM'] + ref_stats)

    # Calculate for each compared DTM
    for idx, arr in enumerate(dtm_c_list):
        # Exclude nodata/inf for stats
        valid = mask & np.isfinite(arr) & np.isfinite(ref_dtm_c)
        diff = arr - ref_dtm_c
        diff = np.where(valid, diff, np.nan)
        # Use valid mask for stats
        if np.count_nonzero(valid) == 0:
            rmse = mae = ssim_with_ref = mean_slope = std_slope = mean_rough = std_rough = ssim_with_img = np.nan
        else:
            rmse = np.sqrt(np.nanmean(diff[valid]**2))
            mae = np.nanmean(np.abs(diff[valid]))
            arr_n = normalise(arr[valid])
            ref_n = normalise(ref_dtm_c[valid])
            try:
                ssim_with_ref = ssim(arr_n, ref_n, data_range=1.0, win_size=21)
            except Exception:
                ssim_with_ref = np.nan
            slope = calc_slope(arr, target_res)
            rough = local_std(arr, 5)
            mean_slope = np.nanmean(slope[valid])
            std_slope = np.nanstd(slope[valid])
            mean_rough = np.nanmean(rough[valid])
            std_rough = np.nanstd(rough[valid])
            arr_m = normalise(arr[valid])
            sm_img_m = normalise(smoothed_image[valid])
            try:
                ssim_with_img = ssim(arr_m, sm_img_m, data_range=1.0, win_size=21)
            except Exception:
                ssim_with_img = np.nan

        stats_results.append([
            os.path.basename(compare_dtm_paths[idx]), rmse, mae, ssim_with_ref,
            mean_slope, std_slope, mean_rough, std_rough, ssim_with_img
        ])
        print(f"Statistics calculated for {os.path.basename(compare_dtm_paths[idx])}.")

    means = ['MEAN']
    for i in range(1, len(stat_names)+1):
        vals = [row[i] for row in stats_results[1:]]
        vals = [v for v in vals if np.isfinite(v)]
        means.append(np.nanmean(vals) if vals else np.nan)
    stats_results.append(means)

    statfile = os.path.join(output_dir, 'statistics.txt')
    with open(statfile, 'w') as f:
        f.write('Filename\t' + '\t'.join(stat_names) + '\n')
        for row in stats_results:
            f.write('\t'.join([str(x) for x in row]) + '\n')
    print(f"Statistics file saved: {statfile}")

    # --- FIGURES ---
    print("Selecting representative patches and creating figures ...")
    PATCH_SZ = 200
    def pick_patch_coords(roughness_map, high=True, n=2):
        valid = np.isfinite(roughness_map)
        patch_means = []
        coords = []
        for i in range(0, roughness_map.shape[0]-PATCH_SZ, PATCH_SZ//2):
            for j in range(0, roughness_map.shape[1]-PATCH_SZ, PATCH_SZ//2):
                patch = roughness_map[i:i+PATCH_SZ, j:j+PATCH_SZ]
                patch_mask = valid[i:i+PATCH_SZ, j:j+PATCH_SZ]
                if patch.shape == (PATCH_SZ, PATCH_SZ) and np.count_nonzero(patch_mask) > 0.95*PATCH_SZ**2:
                    patch_mean = np.nanmean(patch)
                    patch_means.append(patch_mean)
                    coords.append((i, j))
        if not patch_means:
            return []
        indices = np.argsort(patch_means)
        chosen = []
        if high:
            for idx in indices[::-1][:n]:
                chosen.append(coords[idx])
        else:
            for idx in indices[:n]:
                chosen.append(coords[idx])
        return chosen

    def plot_patch(ref_img, ref_dtm, tgt_dtm, i, j, patch_idx, dtm_name, outdir):
        ref_img_disp = exposure.equalize_hist(ref_img)
        vmin = np.nanpercentile(ref_dtm, 2)
        vmax = np.nanpercentile(ref_dtm, 98)
        ref_dtm_disp = np.clip(ref_dtm, vmin, vmax)
        tgt_dtm_disp = np.clip(tgt_dtm, vmin, vmax)
        def hillshade(arr, azimuth=315, angle_altitude=45):
            x, y = np.gradient(arr, target_res[0], target_res[1])
            slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
            aspect = np.arctan2(-x, y)
            az = np.deg2rad(azimuth)
            alt = np.deg2rad(angle_altitude)
            shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
            return (shaded - shaded.min()) / (shaded.max() - shaded.min())
        tgt_hs = hillshade(tgt_dtm_disp)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(ref_img_disp, cmap='gray')
        axes[0].set_title('Reference Image')
        axes[1].imshow(ref_dtm_disp, cmap='viridis')
        axes[1].set_title('Reference DTM')
        axes[2].imshow(tgt_dtm_disp, cmap='viridis')
        axes[2].set_title(f'Target DTM')
        axes[3].imshow(tgt_hs, cmap='gray')
        axes[3].set_title('Target DTM Hillshade')
        for ax in axes:
            ax.axis('off')
        plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04, orientation='horizontal')
        plt.colorbar(axes[2].images[0], ax=axes[2], fraction=0.046, pad=0.04, orientation='horizontal')
        # Create a separate scalebar for each axes (to avoid reusing the artist)
        for ax in axes:
            scalebar = ScaleBar(target_res[0], units='m', location='lower left', box_alpha=0.3, length_fraction=0.25)
            ax.add_artist(scalebar)
        plt.tight_layout()
        outfn = os.path.join(outdir, f'{dtm_name}_patch_{patch_idx:02d}_{i}_{j}.png')
        plt.savefig(outfn, dpi=150)
        plt.close(fig)

    for dtm_idx, (tgt_dtm, tgt_path) in enumerate(zip(dtm_c_list, compare_dtm_paths)):
        tgt_rough = local_std(tgt_dtm, 5)
        coords = pick_patch_coords(tgt_rough, high=True, n=2) + pick_patch_coords(tgt_rough, high=False, n=2)
        dtm_name = os.path.splitext(os.path.basename(tgt_path))[0]
        patch_num = 0
        for (i, j) in coords:
            ref_img_patch = image_c[i:i+PATCH_SZ, j:j+PATCH_SZ]
            ref_dtm_patch = ref_dtm_c[i:i+PATCH_SZ, j:j+PATCH_SZ]
            tgt_dtm_patch = tgt_dtm[i:i+PATCH_SZ, j:j+PATCH_SZ]
            # Check valid region for the patch
            patch_mask = mask[i:i+PATCH_SZ, j:j+PATCH_SZ]
            if ref_img_patch.shape == (PATCH_SZ, PATCH_SZ) and np.count_nonzero(patch_mask) > 0.95 * PATCH_SZ ** 2:
                plot_patch(ref_img_patch, ref_dtm_patch, tgt_dtm_patch, i, j, patch_num, dtm_name, output_dir)
                patch_num += 1
        print(f"Figures for {dtm_name} completed.")

    print("All done! Quick validation completed.")

if __name__ == "__main__":
    main()

