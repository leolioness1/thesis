import json
import os

import rasterio
import rasterio.mask
from rasterio.windows import Window
from rasterio.features import shapes

import numpy as np
import geopandas as gpd


def check_crs(input_file):
    f = rasterio.open(input_file)
    return f.crs


def convert_dtype(input_file, dtype):    
    with rasterio.open(input_file) as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'dtype': rasterio.uint16,
        })

        with rasterio.open(input_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i).astype(dtype), i)


def polygon_to_raster_coords(poly, poly_crs, target_crs=None):
    gdf = gpd.GeoDataFrame({'geometry': poly}, index=[0], crs=poly_crs)
    if target_crs:
        gdf = gdf.to_crs(target_crs)
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def clip_rasters(layers, coords, output_path='/tmp', names=None):
    clipped = []
    for i, layer in enumerate(layers):
        with rasterio.open(layer) as src:
            out_img, out_transform = rasterio.mask.mask(src, coords, crop=True)
            out_meta = src.meta
            out_meta.update({
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform
            })

        if names:
            name = names[i]
        else:
            name = os.path.splitext(os.path.basename(layer))[0]
    
        output_file = output_path + '/' + name + '_clipped.tiff'

        with rasterio.open(output_file, "w", **out_meta) as dst:
            dst.write(out_img)

        clipped.append(output_file)

    return clipped


def polygonise(predictions, threshold, crs, transform):
    # Threshold the predictions and convert to binary integer representation
    thresholded = (predictions > threshold).astype(np.uint8)

    geometries = [
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v)
        in enumerate(shapes(thresholded, connectivity=8, transform=transform))
    ]

    gdf = gpd.GeoDataFrame.from_features(geometries, crs=crs)
    gdf = gdf[gdf['raster_val']==1]

    return gdf


def read_sub_arrays(img, window_x, window_y):
    sub_arrs = []
    for i in range(img.shape[0] // window_x):
        for j in range(img.shape[1] // window_y):
            sub_arrs.append(img.read(window=Window(i * window_x, j * window_y, window_x, window_y)))

    return sub_arrs


def create_training_arrays(dataset, window,name, out_meta):
    subs = read_sub_arrays(dataset, window, window)
    out = []

    for i,sub in enumerate(subs):
        print(i)
        if (sub[1] == 1).sum() != 0 and (sub[4:8]!=0).sum()!=0:
            output_file = fr'C:\Users\leo__\PycharmProjects\Perma_Thesis\output_windows\{name}_window_{i}.tiff'

            with rasterio.open(output_file, "w",**out_meta) as dst:
                dst.write(sub[1:, :, :])
            out.append(sub[1:, :, :])
    return out


