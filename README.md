# Swap-Mukham
[![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-blue?logo=google-colab&logoColor=white)](https://colab.research.google.com/github/harisreedhar/Swap-Mukham/blob/main/swap_mukham_colab.ipynb)
## Description

A simple face swapper based on insightface inswapper heavily inspired by roop.

![swapmukham_faceenhanced](https://github.com/harisreedhar/Swap-Mukham/assets/46858047/c0c34eac-6b48-4c2f-9222-a85e8fb76b43)


## Features
- Easy to use Gradio gui
- Support Image, Video, Directory inputs
- Swap specific face (face recognition)
- Video trim tool
- Face enhancer (GFPGAN)
- Face parsing mask
- colab support

## Installation
### CPU Install
````
git clone https://github.com/harisreedhar/Swap-Mukham
cd Swap-Mukham
conda create -n swapmukham python=3.10 -y
conda activate swapmukham
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_cpu.txt
python app.py
````
### GPU Install (CUDA)
````
git clone https://github.com/harisreedhar/Swap-Mukham
cd Swap-Mukham
conda create -n swapmukham python=3.10 -y
conda activate swapmukham
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
python app.py --cuda
````
## Downolad Models
- [inswapper_128.onnx](https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx)
- [GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth)
- [79999_iter.pth](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812)
- place these models inside ``/assets/pretrained_models/``

## Disclaimer

We would like to emphasize that our deep fake software is intended for responsible and ethical use only. We must stress that **users are solely responsible for their actions when using our software**.

1. Intended Usage:
Our deep fake software is designed to assist users in creating realistic and entertaining content, such as movies, visual effects, virtual reality experiences, and other creative applications. We encourage users to explore these possibilities within the boundaries of legality, ethical considerations, and respect for others' privacy.

2. Ethical Guidelines:
Users are expected to adhere to a set of ethical guidelines when using our software. These guidelines include, but are not limited to:

Not creating or sharing deep fake content that could harm, defame, or harass individuals.
Obtaining proper consent and permissions from individuals featured in the content before using their likeness.
Avoiding the use of deep fake technology for deceptive purposes, including misinformation or malicious intent.
Respecting and abiding by applicable laws, regulations, and copyright restrictions.

3. Privacy and Consent:
Users are responsible for ensuring that they have the necessary permissions and consents from individuals whose likeness they intend to use in their deep fake creations. We strongly discourage the creation of deep fake content without explicit consent, particularly if it involves non-consensual or private content. It is essential to respect the privacy and dignity of all individuals involved.

4. Legal Considerations:
Users must understand and comply with all relevant local, regional, and international laws pertaining to deep fake technology. This includes laws related to privacy, defamation, intellectual property rights, and other relevant legislation. Users should consult legal professionals if they have any doubts regarding the legal implications of their deep fake creations.

5. Liability and Responsibility:
We, as the creators and providers of the deep fake software, cannot be held responsible for the actions or consequences resulting from the usage of our software. Users assume full liability and responsibility for any misuse, unintended effects, or abusive behavior associated with the deep fake content they create.

By using our deep fake software, users acknowledge that they have read, understood, and agreed to abide by the above guidelines and disclaimers. We strongly encourage users to approach deep fake technology with caution, integrity, and respect for the well-being and rights of others.

Remember, technology should be used to empower and inspire, not to harm or deceive. Let's strive for ethical and responsible use of deep fake technology for the betterment of society.


## Acknowledgements

- [Roop](https://github.com/s0md3v/roop)
- [Insightface](https://github.com/deepinsight)
- [Ffmpeg](https://ffmpeg.org/)
- [Gradio](https://gradio.app/)
- [Wav2lip HQ](https://github.com/Markfryazino/wav2lip-hq)
- [Face Parsing](https://github.com/zllrunning/face-parsing.PyTorch)

## Loved my work?
[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/harisreedhar)

## License

[MIT](https://choosealicense.com/licenses/mit/)
