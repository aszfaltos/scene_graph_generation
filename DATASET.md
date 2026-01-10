# DATASET

## Visual Genome
The following is adapted from [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs).

Note that our codebase intends to support attribute-head too, so our ```VG-SGG.h5``` and ```VG-SGG-dicts.json``` are different with their original versions in [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs). We add attribute information and rename them to be ```VG-SGG-with-attri.h5``` and ```VG-SGG-dicts-with-attri.json```. The code we use to generate them is located at ```datasets/vg/generate_attribute_labels.py```. Although, we encourage later researchers to explore the value of attribute features, in our paper "Unbiased Scene Graph Generation from Biased Training", we follow the conventional setting to turn off the attribute head in both detector pretraining part and relationship prediction part for fair comparison, so does the default setting of this codebase.

### Download:
1. Download the VG images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip).
2. Download the [scene graphs](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EfI9vkdunDpCqp8ooxoHhloBE6KDuztZDWQM_Sbsw_1x5A?e=fjTSvw) and extract them.

### Setup:

The code expects files at these locations:
- Images: `datasets/vg/stanford_spilt/VG_100k_images/`
- H5 file: `datasets/vg/VG-SGG-with-attri.h5`
- Dict file: `datasets/vg/VG-SGG-dicts-with-attri.json`
- Image metadata: `datasets/vg/image_data.json`
- Zero-shot triplets (for evaluation): `pysgg/data/datasets/evaluation/vg/zeroshot_triplet.pytorch`

#### Option A: Direct placement (recommended)

Extract and move files directly to the expected locations:
```bash
# Create directory structure
mkdir -p datasets/vg/stanford_spilt

# Move images (combine both parts into one folder)
mv <path-to-extracted>/VG_100K datasets/vg/stanford_spilt/VG_100k_images
# If you have VG_100K_2 separately (use find to handle many files):
find <path-to-extracted>/VG_100K_2 -type f -exec mv {} datasets/vg/stanford_spilt/VG_100k_images/ \;

# Move annotation files
mv <path-to-extracted>/VG-SGG-with-attri.h5 datasets/vg/
mv <path-to-extracted>/VG-SGG-dicts-with-attri.json datasets/vg/
mv <path-to-extracted>/image_data.json datasets/vg/

# Setup zero-shot evaluation file
mkdir -p pysgg/data/datasets/evaluation/vg
cp <path-to-extracted>/zeroshot_triplet.pytorch pysgg/data/datasets/evaluation/vg/
```

#### Option B: Symlinks

If you want to keep files in a separate location:
```bash
mkdir -p datasets/vg/stanford_spilt

# Link images
ln -s /path-to-vg/VG_100K datasets/vg/stanford_spilt/VG_100k_images

# Link annotation files (these go directly in vg/, NOT in stanford_spilt/)
ln -s /path-to-vg/VG-SGG-with-attri.h5 datasets/vg/VG-SGG-with-attri.h5
ln -s /path-to-vg/VG-SGG-dicts-with-attri.json datasets/vg/VG-SGG-dicts-with-attri.json
ln -s /path-to-vg/image_data.json datasets/vg/image_data.json

# Zero-shot evaluation file
mkdir -p pysgg/data/datasets/evaluation/vg
cp /path-to-vg/zeroshot_triplet.pytorch pysgg/data/datasets/evaluation/vg/
```

**Note:** If you want to use different directories, edit the paths in `DATASETS['VG_stanford_filtered_with_attribute']` in `pysgg/config/paths_catalog.py`.

## Openimage V4/V6 

### Download
The initial dataset(oidv6/v4-train/test/validation-annotations-vrd.csv) can be downloaded from [offical website]( https://storage.googleapis.com/openimages/web/download.html).
The Openimage is a very large dataset, however, most of images doesn't have relationship annotations. 
To this end, we filter those non-relationship annotations and obtain the subset of dataset ([.ipynb for processing](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EebESIOrpR5NrOYgQXU5PREBPR9EAxcVmgzsTDiWA1BQ8w?e=46iDwn) ). 
You can download the processed dataset: [Openimage V6(38GB)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EXdZWvR_vrpNmQVvubG7vhABbdmeKKzX6PJFlIdrCS80vw?e=uQREX3),
[Openimage V4(28GB)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EVWy0xJRx8RNo-zHF5bdANMBTYt6NvAaA59U32o426bRqw?e=6ygqFR)
The dataset dir contains the `images` and `annotations` folder. Link the `open_image_v4` and `open_image_v6` dir to the `/datasets/openimages` then you are ready to go.
