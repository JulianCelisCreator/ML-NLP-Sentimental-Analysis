# ml_npl

Python package for the Team 4 sentiment-analysis project. See the
[repository README](../README.md) for setup, layout and project status.

```python
from ml_npl.dataset import load_reviews
from ml_npl import eda, plots

reviews = load_reviews()                                   # 42,656 validated rows
labels = eda.sentiment_from_rating(reviews["Rating"])      # the target
```
