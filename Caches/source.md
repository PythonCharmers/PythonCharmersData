# scikit-learn dataset caches

`data_home="/Data/Caches"` in the course notes points scikit-learn's dataset
fetchers here. On a JupyterHub /Data is read-only for participants, so a
fetcher that finds no cache tries to download into it and fails; these files
are what lets the cells run there.

- `cal_housing_py3.pkz`: written by `sklearn.datasets.fetch_california_housing`.
  California housing, 1990 US census block groups, from the StatLib repository
  (Pace, R. Kelley and Ronald Barry, "Sparse Spatial Autoregressions",
  Statistics and Probability Letters, 33 (1997) 291-297). scikit-learn
  downloads it from its figshare mirror. No explicit licence is stated.
  File dated 2026-03-10.
- `20news-bydate_py3.pkz`: written by `sklearn.datasets.fetch_20newsgroups`.
  The "bydate" version of the 20 Newsgroups corpus (Ken Lang), as distributed
  at <http://qwone.com/~jason/20Newsgroups/>; scikit-learn downloads it from
  its figshare mirror. No explicit licence is stated. File dated 2026-07-27.
- Both are compressed pickles in scikit-learn's own cache format. The notes
  were checked against scikit-learn 1.9.1.
