# Data

Every file below is listed with its md5 in `data/MD5SUMS`. The training data are what trained the released heads. The
scripts in `scripts/` rebuild them from the corpus (see the README).

## Corpus: `data/corpus/corpus_elite_ccf_core_thu.jsonl.gz`

- **Records:** 607,416 OpenAlex works, one JSON object per line (md5 dd3890ab3c83791a1bce4dc700e5516a).
- **Fields:** `paper_id` (OpenAlex id), `title`, `abstract`, `authors` (OpenAlex author ids in byline order), `year`,
  `concept`, `venue_key`, `venue_name`, `ccf_tier`, `core_tier` and `thu_tier`.
- **Venues:** 314 computer-science venues ranked CCF A or B, CORE A* or A, or THU A or B.
- **Decontamination:** applied before any training data was built.
  - It removed 34,509 records that match a paper of any of the seven benchmarks, either a submission or a paper in a
    reviewer's history. A record matches on its normalized title (at least 15 characters and 3 words), on the first 120
    characters of its abstract, or on the OpenAlex id of a KDD paper.
  - It removed the 1,714 OpenAlex author ids linked to benchmark reviewers from every author list (62,234 occurrences).
  - The corpus went from 641,925 to 607,416 records.
- **License:** OpenAlex data are released under CC0.

## Training and validation episodes: `data/train/`

`scripts/build_training_data.py` builds them from the corpus. The steps are listed in its docstring.

**Files**
- `episodes.npz`: the episode arrays (described in `muster/data.py`).
- `works.tsv`: the OpenAlex id and the corpus line of each of the 446,314 rows.
- `vectors_sp2.npy`, `vectors_sci.npy`: (446314, 768) float32 unit vectors.
- `mu_sp2.npy`, `mu_sci.npy`: the mean of the 431,972 rows that belong to training authors' records (the corpus mean
  of the adapter).

**Authors and episodes**
- There are 30,000 training authors and 2,000 validation authors, the 32,000 most prolific authors with at least 6
  papers. Each keeps its newest 49 papers.
- A paper credits its first four authors.
- A training author gives three episodes, from its three newest papers. Each query's history holds the author's up to
  48 previous papers, newest first: on average 21.4 papers, at least 8, and 48 in 2.9% of episodes.
- There are 74,077 distinct query papers.
- A validation author gives one episode, from its newest paper and its 10 to 48 previous papers (22.1 on average).

**Caveats**
- Papers are ordered by integer year, then by corpus line. So "earlier" holds only at the granularity of a year: 5.3% of
  history papers share the query's year.
- The validation authors differ from the training authors, but many of their papers are co-authored with training
  authors. This holds for 57.8% of validation queries and 68.0% of validation history papers.

## Long-history validation: `data/longval/`

- **Authors:** the same 2,000 validation authors and queries, with each author's full earlier record (up to 256
  papers: median 18, maximum 197, 8.5% longer than 48).
- **Rows:** 48,741 works, listed in `works.tsv` with their training row. 1,390 works are not among the training rows
  and were encoded separately.
- **Vectors:** `vectors_<enc>.npy` holds the training vectors for the works that are training rows.

## Frozen vectors

The encoders are `allenai/specter2_base` (revision 3447645e1def9117997203454fa4495937bfbd83) and `malteos/scincl`
(revision ebc5348d184ba2fc9beee69b4e394263fce57b2e). Both encode in FP32 with at most 512 tokens.

**Corpus works**
- Text: "title [SEP] abstract".
- Pooling: multi-view pooling of the top layers (see `scripts/encode.py`).
- Encoded on a GPU with TF32 off.

**Benchmark papers**
- Pooling: the last layer's [CLS] vector, unit-normalized.
- SPECTER2: "title. abstract", from the shared caches that every method in the paper's tables used. They were encoded
  under BF16 autocast.
- SciNCL: "title[SEP]abstract" for LR-Bench and CMU (stored in FP16), and "title [SEP] abstract" for the ranking
  benchmarks.
- In the SciNCL ranking matrices, papers that no benchmark item uses are zero rows.

**Re-encoding**
- `scripts/encode.py check` re-encodes a random sample of corpus and benchmark papers and reports cosine similarity against the precomputed vectors.

## Negative screen: `data/screen/`

**`references.jsonl.gz`**
- The OpenAlex reference lists of the 446,314 rows, fetched on 28 Oct 2026.
- OpenAlex returned 383,390 works. 338,841 of them list at least one reference, and the lists name 2,074,414 distinct
  works.
- `refs_csr.npz` is the same as a sparse row-by-reference index (`scripts/build_references.py`).

**`model_<enc>.json`**
- The fitted latent-class model of each encoder (`scripts/fit_screen.py`).
- Text threshold t_T: 0.9065 for SPECTER2 and 0.8445 for SciNCL. Citation threshold t_G: 0.0159.
- History-length group cuts: 14 and 23 papers.
- The posterior table is indexed by fold, group, authorship, text and citation evidence.
- Training keeps a non-author as a negative if its posterior is at most 0.2. Reference lists are used only in training.

## Benchmarks

See [../benchmarks/README.md](../benchmarks/README.md).
