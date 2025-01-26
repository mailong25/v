# Real-Time Video-Based Chatbot: Digital Immortality of a Real Person

## **Overview**
This project aims to create a **real-time video-based chatbot** that embodies the digital immortality of a real person. The chatbot will:  
- Possess a **predefined persona and life story**.  
- Behave naturally, with:  
  - **Realistic voices**.  
  - **Natural facial expressions**.  
- Adapt and grow over time.  
- Be **useful**, scalable, and capable of natural, human-like interactions.

---

## **Key Features**
- **Results-Oriented**: The focus is on achieving the desired goals, whether using custom-built code or publicly available solutions.  
- **Local Deployment**:  
  - All models should ideally run locally for **lower latency** and **greater control**.  
  - **Fast data transfer** is essential to enable real-time interaction.  
- **Modular Design**: Core functionalities are divided into clear, distinct modules for flexibility and scalability.  

---

## **Development Roadmap**
### **Version 1: Text-Based Turn-by-Turn Interaction**
- **Modules**:  
  - ASR (Automatic Speech Recognition) – API-based.  
  - Response Generation – Local (using LLMs).  
  - TTS (Text-to-Speech) – API-based.  
  - Talking Head Generation – Local.  
- **Features**:  
  - Basic turn-by-turn interaction with a focus on coherent and timely responses.  

### **Version 2: Text-Based with Turn-Taking Prediction**
- **Modules**:  
  - ASR – Fully local.  
  - Turn-Taking & Backchannel Prediction – Predict when the bot/user should speak, with natural pauses, fillers, and acknowledgments (e.g., "uh-hum").  
  - Natural Response Generation – Enhanced for fluidity.  
  - Natural TTS – Produce lifelike speech.  
  - Natural Talking Head Generation – Add subtle and realistic facial animations.  
- **Features**:  
  - Smoother conversational flow with turn-taking.  
  - Mimics human conversation dynamics.  

### **Version 3: Fully Real-Time with Speech-to-Speech Interaction**
- **Modules**:  
  - Dialog Manager – Manages speaking, stopping, continuing, or remaining silent.  
  - Speech Response Generation – Direct Speech-to-Speech models for real-time interaction.  
  - Talking Head Generation – Use advanced models like Audio2Face for highly natural visual responses.  
- **Features**:  
  - End-to-end real-time interaction with natural flow.  
  - Minimal latency and highly adaptive responses.

---

## **Core Modules**
1. **Streaming ASR**:  
   - Converts real-time speech input into text.  
2. **Turn-Taking Prediction**:  
   - Predicts silence, speaking turns, and continuation signals.  
3. **Response Generation**:  
   - Generates human-like text or speech responses.  
4. **Text-to-Speech (TTS)**:  
   - Produces natural and expressive speech from text.  
5. **Talking Head Generation**:  
   - Creates a realistic video representation with synchronized facial expressions and lip movements.  

---

## **Input/Output Design**
### **Input**:  
- **Streaming audio** (speech input).  
- **Streaming text** (typed or transcribed).  
- **Timestamps**:  
  - Speaker label (e.g., Spk1, Spk2).  
  - Content timestamps for alignment.  
- **Non-linear context**:  
  - Context input (e.g., Speaker 1 → Speaker 2, etc.) can leverage ChatGPT for tracking and managing.  

### **Processing**:  
- **Turn-Taking**:  
  - Handles silent, speaking, continuation, and stopping cues.  
- **Streaming Input**:  
  - Supports both speech-only and text-only inputs.  

### **Output**:  
- **Video Response**:  
  - A synchronized video with audio (face and voice) generated for the chatbot's response.

---

## **Enhancements and Considerations**
1. **S2S Model Improvements**:  
   - Fine-tune existing Speech-to-Speech models to produce more natural results.  
   - Redesign prompts to encourage lifelike responses.  

2. **Training Goals**:  
   - Focus on fine-tuning models for expressive TTS and nuanced response generation.  
   - Prioritize speed, adaptability, and scalability.  

3. **Scalability**:  
   - Ensure the architecture supports future growth and enhancements.  

---

## **Getting Started**

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers
pip install librosa scipy tensorflow opencv-python
conda install opencv libsndfile

Install apex

wget https://storage.googleapis.com/mailong25/wav2lip/face_det_results_1min.pkl
wget https://storage.googleapis.com/mailong25/wav2lip/1min.mp4
wget https://storage.googleapis.com/mailong25/wav2lip/wav2lip_gan.pth
```

