# Simple Face Anti-Spoofing App

The security system based on face recognition is increasingly improving due to the convenience, accuracy, and reliability it provides. 
Simultaneously, there are efforts to exploit weaknesses in the system through face presentation attacks using face recognition technology. 
The approach of face liveness detection or face anti-spoofing is used to address face presentation attacks. In this project, we aim to develop 
machine learning model to differentiate between real and spoofed face biometric presentation. 

The model was trained on [Replay Attack Database](https://www.idiap.ch/en/scientific-research/data/replayattack) and has been submitted for my
[undergraduate final thesis](https://etd.repository.ugm.ac.id/penelitian/detail/225405). This project was based on Tensorflow VGG-16 pre-trained model with modification on its output layer. 

## Deployed App

* Tech stack: OpenCV, NumPy, Pillow, Tensorflow Lite, Streamlit.
* Running app: https://face-anti-spoofing-app.streamlit.app/

## Demo

**Prediction On Spoofed Face Biometric Representation**

![image](https://github.com/saskia-dwi-ulfah/face-anti-spoofing-app/assets/73946560/7739bcfb-fb5e-4e34-8bf6-fe5c626aac9b)

**Prediction On Real Face Biometric Representation**

![image](https://github.com/saskia-dwi-ulfah/face-anti-spoofing-app/assets/73946560/4ff7bbee-38ee-433d-ad96-81ca4143c082)


