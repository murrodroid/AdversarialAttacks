# AdversarialAttacks

## Prerequisites

- Python 3.8 or higher (recommends 3.11)

## Easy Setup w CONDA (ONLY CUDA-compatible)

1. Execute the environment-file
   ```sh
   conda env create -f environment.yml
   ```

## Alternative Setup

1. Clone the repository
   ```sh
   git clone https://github.com/murrodroid/AdversarialAttacks
   cd AdversarialAttacks
   ```

2. Install PyTorch with CUDA support
   ```sh
   pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

3. Install remaining dependencies
   ```sh
   pip install -r requirements.txt
   ```