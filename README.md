
**BioHybrid-GS** is a deep learning framework designed to enhance Genomic Prediction (GP) accuracy for complex traits (e.g., cold tolerance in rice) by integrating biological priors. 

Unlike traditional GBLUP or standard deep learning models, this **Bio-Hybrid Transformer Ensemble** employs a dual-stream architecture:
1.  **Background Stream:** Captures polygenic background effects using genome-wide markers.
2.  **Bio-Prior Stream:** Captures large-effect loci using a Transformer encoder focused on biologically significant markers derived from **GWAS**, **Differential Alternative Splicing (DAS)**, and **WGCNA Hub Genes**.

This repository contains the implementation described in our manuscript.

![Uploading image.png…]()


## Key Features

* **Dual-Stream Architecture:** Effectively balances genome-wide background noise and trait-specific signals.
* **Ensemble Learning:** Uses a voting mechanism (default 5 runs) to ensure robust predictions.
* **Saliency Map Analysis:** Includes a built-in interpretability module to identify top functional SNPs (saved as CSV and Manhattan plots).
* **Optimized for Stability:** Tuned for CPU/MPS execution to ensure reproducibility across different hardware.

## System Requirements

* **OS:** Linux / macOS / Windows
* **Python:** 3.9 or higher
* **Hardware:** 16GB RAM recommended. The code is optimized for CPU efficiency but supports CUDA/MPS if configured.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Haozhen/BioHybrid-GS.git](https://github.com/Haozhen/BioHybrid-GS.git)
   cd BioHybrid-GS
