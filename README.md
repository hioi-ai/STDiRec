STDiRec is a sequential recommendation framework that integrates coarse-grained user interests and fine-grained directional transition patterns through a dual-path diffusion architecture.
```
STDiRec/
├── datasets/           # Dataset files
├── scripts/            # Training scripts
│   ├── beauty.sh
│   ├── Sports.sh
│   └── toys.sh
├── datasets.py         # Data loading and preprocessing
├── main.py             # Main entry point
├── models.py           # Recommendation model
├── modules.py          # Model components
├── trainer.py          # Training and evaluation
└── utils.py            # Utility functions
```
Experiments are conducted on the following Amazon datasets:
- Beauty
- Sports and Outdoors
- Toys and Games
Training scripts are provided in the scripts/ directory.
For example:
```
bash scripts/beauty.sh
```
The implementation of STDiRec is available in this repository. Additional documentation will be updated progressively.
