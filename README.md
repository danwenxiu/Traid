# TriadFormer: Unifying State Space Modeling and Anatomical Transformers for Structured 3D Human Pose Estimation


---

## ⚙️ Dependencies

- Python 3.10.9  
- PyTorch 1.12.1  

### Install

```bash
pip install -r requirements.txt



## Test the model

To test on Human3.6M with 2D poses detected by CPN as inputs, run:

python main.py --reload --keypoints cpn_ft_h36m_dbb --previous_dir "ckpt/cpn"

To test on Human3.6M with GT 2D poses as inputs, run:

python main.py --reload --keypoints gt --previous_dir "ckpt/gt" 



Our code refers to the following repositories.
- [ DC-GCT](https://github.com/KHB1698/DC-GCT)
- [Mamba](https://github.com/state-spaces/mamba)
- [PoseSSM](https://github.com/yinhanxi/PoseSSM)
  
We recommend constructing the demo by referring to [GraphMLP](https://github.com/Vegetebird/GraphMLP), and you can achieve it by replacing the GraphMLP model.
We thank the authors for releasing their codes.
This is the official repository for our paper. The code will be released upon paper acceptance.
