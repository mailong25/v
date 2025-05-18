# Real-Time Video-Based Chatbot

This repository contains the implementation of the first real-time video-based chatbot, developed in 2020—way before the rise of generative AI.

A demo video is available on [YouTube](https://www.youtube.com/watch?v=iXM8B1qHcD4).

## **Overview**
This project aims to create a **real-time video-based chatbot** that embodies the digital immortality of a real person. The chatbot will:  
- Possess a **predefined persona and life story**.  
- Behave naturally, with:  
  - **Realistic voices**.  
  - **Natural facial expressions**.  
- Adapt and grow over time.  
- Be **useful**, scalable, and capable of natural, human-like interactions.

---

## **Current Version: Text-Based Turn-by-Turn Interaction**
- **Modules**:  
  - ASR (Automatic Speech Recognition) – API-based.  
  - Response Generation – Local (using Blenderbot).  
  - TTS (Text-to-Speech) – API-based.  
  - Talking Head Generation – Local (using Wav2Lip)
- **Features**:  
  - Basic turn-by-turn interaction with a focus on coherent and timely responses.  

## **Future Development Roadmap**

### **Version 2: Fully Real-Time with Speech-to-Speech Interaction**
- **Modules**:  
  - Dialog Manager – Manages speaking, stopping, continuing, or remaining silent.  
  - Speech Response Generation – Direct Speech-to-Speech models for real-time interaction.  
  - Talking Head Generation – Use advanced models like Audio2Face for highly natural visual responses.  
- **Features**:  
  - End-to-end real-time interaction with natural flow.  
  - Minimal latency and highly adaptive responses.

More details can be found in our recent paper [Real-time textless dialogue generation](https://arxiv.org/abs/2501.04877) and [implementation](https://github.com/mailong25/rts2s-dg).

