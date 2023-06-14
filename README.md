# Swap-Mukham

## Description

A simple face swapper based on insightface inswapper heavily inspired by roop.

![preview](https://github.com/harisreedhar/Swap-Mukham/assets/46858047/354aeeba-0b30-48cd-b6d9-1e359e3940e2)


## Installation and Usage

1. Clone repository:
- ``git clone https://github.com/harisreedhar/Swap-Mukham``
- ``cd Swap-Mukham``


2. Download model from the link and place it in root directory.
- ``https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx``


3. Create conda environment (Assuming anaconda is already installed):
- ``conda create -n swapmukham python=3.10 -y``
- ``conda activate swapmukham``


4. Install ffmpeg:
- ``conda install -c conda-forge ffmpeg``


5. Install requirements:
- ``pip install -r requirements.txt``


6. Run the Application:
- ``python app.py``


7. Access the Application:
- Once the application is running, a port number will be displayed, such as http://127.0.0.1:7860/. Open this URL in your browser.


8. Dark Theme:
- If you prefer a dark theme, append ``/?__theme=dark`` to the end of the URL.


## Disclaimer

**Disclaimer:**

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

- [https://github.com/s0md3v/roop](Roop)
- [https://github.com/deepinsight](Insightface)
- [https://ffmpeg.org/](Ffmpeg)
- [https://gradio.app/](Gradio)

## Loved my work?
[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/harisreedhar)

## License

[MIT](https://choosealicense.com/licenses/mit/)
