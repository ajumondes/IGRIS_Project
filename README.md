# IGRIS - Intelligent Genuine-user Recognition and Identification System

IGRIS is a continuous user authentication system based on behavioral biometrics. It aims to transparently authenticate a user throughout their session by analyzing their unique keyboard typing patterns using an LSTM Autoencoder.

## 🚀 Project Status
This project is an advanced prototype. The core end-to-end system is working, using a deep learning model for anomaly detection.

## ✨ Features
* **User Management:** Full user registration, login, and session management.
* **Web-based Enrollment:** Users train a unique LSTM Autoencoder model on their "typing signature" through a web interface.
* **Background Data Collection:** A standalone agent captures system-wide keystroke dynamics in real-time.
* **Real-time Scoring:** The server uses the user's trained model to detect anomalies in live typing data, generating a "trust score."

## 🛠️ Technical Stack
* **Backend:** Python, Flask
* **Database:** SQLAlchemy, SQLite
* **Machine Learning:** TensorFlow (Keras), Scikit-learn, Pandas, NumPy
* **Data Collection:** pynput

---

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/IGRIS-Project.git](https://github.com/YourUsername/IGRIS-Project.git)
cd IGRIS-Project