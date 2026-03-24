# TriadFormer: Unifying State Space Modeling and Anatomical Transformers for Structured 3D Human Pose Estimation
## Dataset Setup
We follow the standard protocol of the Human3.6M dataset for training and evaluation.
Please download the processed dataset files from the official sources or provided links, and organize them as follows:
${POSE_ROOT}/
├── dataset/
│   ├── data_3d_h36m.npz
│   ├── data_2d_h36m_gt.npz
│   └── data_2d_h36m_cpn_ft_h36m_dbb.npz

## Dependencies
Our implementation is based on the following environment:

- Python 3.10.9  
- PyTorch 1.12.1  

Install all required packages via:

```bash
pip install -r requirements.txt


## Test the model
To test on Human3.6M with 2D poses detected by CPN as inputs, run:
```bash
python main.py --reload --keypoints cpn_ft_h36m_dbb --previous_dir "ckpt/cpn"

To test on Human3.6M with GT 2D poses as inputs, run:
```bash
python main.py --reload --keypoints gt --previous_dir "ckpt/gt" 



Our code refers to the following repositories.
- [ DC-GCT](https://github.com/KHB1698/DC-GCT)
- [Mamba](https://github.com/state-spaces/mamba)
- [PoseSSM](https://github.com/yinhanxi/PoseSSM)
  
We recommend constructing the demo by referring to [GraphMLP](https://github.com/Vegetebird/GraphMLP), and you can achieve it by replacing the GraphMLP model.
We thank the authors for releasing their codes.
This is the official repository for our paper. The code will be released upon paper acceptance.
