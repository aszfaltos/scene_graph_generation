# DATASET

## Pretrained Language Models

The semantic encoders require pretrained word embedding models. These should be placed in the `models/` directory. Select the embedding type using the `--embedding_type` argument:

```bash
# Training with different embedding types
uv run python train.py --embedding_type bert    # BERT (768-dim, default)
uv run python train.py --embedding_type minilm  # MiniLM (384-dim)
uv run python train.py --embedding_type word2vec  # Word2Vec (300-dim)
uv run python train.py --embedding_type glove   # GloVe (300-dim, for baselines)
```

### Word2Vec (Google News)

Download the Google News Word2Vec model (300-dimensional, ~1.5GB compressed):

```bash
mkdir -p models/word2vec

# Option 1: Using gensim downloader (recommended, ~1.7GB download)
uv run python -c "
import gensim.downloader as api
model = api.load('word2vec-google-news-300')
model.save_word2vec_format('models/word2vec/GoogleNews-vectors-negative300.bin', binary=True)
print('Word2Vec model saved')
"

# Option 2: Download original from Google Drive using gdown
uv pip install gdown
uv run python -m gdown 0B7XkCwpI5KDYNlNUTTlSS21pQmM -O models/word2vec/GoogleNews-vectors-negative300.bin.gz
gunzip models/word2vec/GoogleNews-vectors-negative300.bin.gz
```

### BERT (bert-base-uncased)

Download the BERT model from Hugging Face (~440MB):

```bash
mkdir -p models/bert

# Option 1: Using uvx hf (recommended)
uvx hf download google-bert/bert-base-uncased --local-dir models/bert

# Option 2: Using Python
uv run python -c "
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
model = AutoModel.from_pretrained('google-bert/bert-base-uncased')
tokenizer.save_pretrained('models/bert')
model.save_pretrained('models/bert')
print('BERT model saved to models/bert')
"
```

### MiniLM (all-MiniLM-L6-v2)

Download the MiniLM sentence transformer model (~90MB):

```bash
mkdir -p models/minilm

# Option 1: Using uvx hf (recommended)
uvx hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/minilm

# Option 2: Using Python (auto-downloads on first use)
uv run python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('models/minilm')
print('MiniLM model saved to models/minilm')
"
```

### GloVe (for baselines)

Download GloVe embeddings (~862MB). Required for baseline models (MotifNet, VCTree, etc.):

```bash
mkdir -p models/glove

# Download and extract GloVe 6B embeddings
curl -L https://nlp.stanford.edu/data/glove.6B.zip -o models/glove/glove.6B.zip
unzip models/glove/glove.6B.zip -d models/glove/
rm models/glove/glove.6B.zip  # Optional: remove zip after extraction
```

This creates `models/glove/glove.6B.{50,100,200,300}d.txt`. The code uses `glove.6B.300d.txt` by default.

---

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

## VRD (Visual Relationship Detection)

The Stanford VRD dataset by Lu et al. (ECCV 2016) - "Visual Relationship Detection with Language Priors".

### Dataset Statistics
- **Images**: 5,000 (4,000 train / 1,000 test)
- **Object categories**: 100
- **Predicate categories**: 70
- **Relationships**: 37,993

### Download

The original Stanford image server is no longer available. Use Kaggle for images and Stanford for annotations:

```bash
mkdir -p datasets/vrd
cd datasets/vrd

# Step 1: Download images from Kaggle (~1.5GB)
# Go to: https://www.kaggle.com/datasets/apoorvshekher/visual-relationship-detection-vrd-dataset
# Download and extract sg_train_images/ and sg_test_images/ to datasets/vrd/
# You can delete the Kaggle annotation files (sg_*_annotations.json) - we don't use them

# Step 2: Download original annotations from Stanford
curl -L https://cs.stanford.edu/people/ranjaykrishna/vrd/json_dataset.zip -o json_dataset.zip
unzip -j json_dataset.zip -d . && rm json_dataset.zip

cd ../..
```

**Why both sources?** The Kaggle dataset contains the images but uses a different annotation format (text-based with ~1000 predicates). The original Stanford annotations use integer indices with the canonical 70 predicates and 100 objects required for the VRD benchmark.

### Expected Directory Structure

```
datasets/vrd/
├── sg_train_images/          # 4,000 training images
│   └── *.jpg
├── sg_test_images/           # 1,000 test images
│   └── *.jpg
├── annotations_train.json    # Training annotations
├── annotations_test.json     # Test annotations
├── objects.json              # 100 object categories
└── predicates.json           # 70 predicate categories
```

### Usage

```bash
# Training on VRD with BGNN model
uv run python train.py --config-file configs/e2e_relBGNN_vrd.yaml

# Or specify datasets directly
uv run python train.py --dataset vrd_train --test_dataset vrd_test

# The dataset identifiers are:
# - vrd_train: Training split (4,000 images)
# - vrd_test: Test split (1,000 images)
```

### Citation

```bibtex
@inproceedings{lu2016visual,
  title={Visual Relationship Detection with Language Priors},
  author={Lu, Cewu and Krishna, Ranjay and Bernstein, Michael and Fei-Fei, Li},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2016}
}
```

---

## Openimage V4/V6

### Download
The initial dataset(oidv6/v4-train/test/validation-annotations-vrd.csv) can be downloaded from [offical website]( https://storage.googleapis.com/openimages/web/download.html).
The Openimage is a very large dataset, however, most of images doesn't have relationship annotations. 
To this end, we filter those non-relationship annotations and obtain the subset of dataset ([.ipynb for processing](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EebESIOrpR5NrOYgQXU5PREBPR9EAxcVmgzsTDiWA1BQ8w?e=46iDwn) ). 
You can download the processed dataset: [Openimage V6(38GB)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EXdZWvR_vrpNmQVvubG7vhABbdmeKKzX6PJFlIdrCS80vw?e=uQREX3),
[Openimage V4(28GB)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EVWy0xJRx8RNo-zHF5bdANMBTYt6NvAaA59U32o426bRqw?e=6ygqFR)
The dataset dir contains the `images` and `annotations` folder. Link the `open_image_v4` and `open_image_v6` dir to the `/datasets/openimages` then you are ready to go.
