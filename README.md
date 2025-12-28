# Breast Histopathology TCGA (TensorFlow + ClickHouse)

Projeto monorepo para classificação de estágio (I–IV) de câncer de mama
em lâminas histopatológicas TCGA-BRCA.

- Modelagem: TensorFlow/Keras (DenseNet/ResNet + ensemble).
- Metadados: ClickHouse (`tcga_slides`, `tcga_patches`).
- Imagens: filesystem local (`data/`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
