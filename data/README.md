# Dataset — Autonomous Vehicle Perception Module

## CIFAR-10 (Auto-Downloads)

CIFAR-10 is automatically downloaded by torchvision when `USE_DEMO = False`.
No manual steps needed — the download happens inside the notebook.

```python
from torchvision import datasets
datasets.CIFAR10(root="./data/raw", download=True)
```

**Size:** ~170 MB | **Images:** 60,000 (32×32 RGB) | **Classes:** 10

---

## Demo Mode (Default)

With `USE_DEMO = True` (the default), no download is needed.
Synthetic images with class-specific color patterns and geometric shapes
are generated programmatically — the full pipeline works immediately.

---

## Real AV Datasets (Future Work — Phase 3)

For production AV systems, consider:

| Dataset | Size | Task |
|---------|------|------|
| [BDD100K](https://bdd-data.berkeley.edu/) | 100K videos | Detection + Segmentation |
| [KITTI](http://www.cvlibs.net/datasets/kitti/) | ~15K images | 3D detection |
| [nuScenes](https://www.nuscenes.org/) | 40K samples | Multi-modal |
| [German Traffic Sign Dataset](https://benchmark.ini.rub.de/gtsrb_news.html) | 50K images | Sign classification |

These require domain-specific annotation tools and hardware beyond the scope of this course project.
