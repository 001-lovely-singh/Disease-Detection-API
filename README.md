# 🩺 CheckUp+

An AI-powered Disease Detection App built using Machine Learning and Rule-Based logic.
The app predicts possible diseases based on user-entered symptoms and provides instant results.

This project was built to combine practical ML implementation, API development, and Android integration into one complete real-world system.

## 📦 Technologies

- `Kotlin`
- `Python`
- `FastAPI`
- `Retrofit`
- `Java`
- `Pandas`
- `Scikit-learn`
- `XGBoost`
- `SMOTE`
- `Pickle`
- `XML`
- `OkHttp`
- `Multi-hot Encoding`
- `Stratified Shuffle Split`
- `Label Encoding`
- `Rule-Based Filtering`
- `XGBClassifier`

## 🦄 Features

Here’s what you can do with MediPredict:

- Select Symptoms
Users can input symptoms manually inside the Android app.

- ML-Based Prediction
The trained ML model predicts the most probable disease based on symptom patterns.

- Rule-Based Filtering
Before showing ML results, a rule-based system filters diseases strictly matching entered symptoms.

- Clean Result Output
Displays full report with name , age and gender with disease names
Rule-based diseases shown line by line
ML prediction shown separately
No confusing percentages (clean UI focus)

- API Integration
The Android app communicates with a FastAPI backend for real-time predictions.

- Fast Response
Optimized backend for quick inference and smooth mobile experience.

## 🎯 How It Works

1. Dataset is cleaned and preprocessed.
2. Symptoms are converted into multi-hot encoded vectors.
3. Data imbalance handled using SMOTE.
4. XGBoost model is trained.
5. Model is saved using Pickle.
6. FastAPI serves the model.
7. Android app sends symptoms → API returns predicted disease.

## 👨‍🍳 The Process

I started by cleaning the dataset and standardizing symptom text.
Then I implemented multi-hot encoding to represent symptoms numerically.

After that:

- Applied SMOTE to balance rare diseases
- Used Stratified Shuffle Split for fair training/testing
- Trained an XGBoost classifier
- Evaluated model accuracy

Next, I built a FastAPI backend to serve predictions.

Finally, I integrated the API with an Android app using Retrofit and displayed clean, structured results.

Testing was done both on:
- Backend API endpoints
- Android result rendering

## 📚 What I Learned

During this project, I've picked up important skills and a better understanding of complex ideas, which improved my logical thinking.

🧠*Machine Learning Pipeline*
- Handling imbalanced datasets
- Feature engineering using multi-hot encoding
- Model evaluation strategies

🔎*Data Cleaning*
- Standardizing symptom strings
- Managing missing values
- Avoiding duplicate patterns

⚙️*Backend Development*
- Building REST APIs with FastAPI
- Handling CORS
- Structuring JSON responses properly

📱*Android Integration*
- Connecting mobile apps with backend APIs
- Managing async calls with Retrofit
- Parsing API responses cleanly

📊*Logical System Design*
- Combining rule-based logic with ML prediction helped me understand:
- When to trust strict logic
- When to use probabilistic models
- How to merge both intelligently

## Running the Project

### 🔹 Option 1: Install APK

1. Download the APK from the link below
2. Enable “Install from Unknown Sources” on your device
3. Install the APK
4. Open the app and start entering symptoms

📦 Download APK:
[Download Here]([your-apk-link](https://drive.google.com/drive/folders/1yMfD2joASSEqcLj8SapalOc0HTaJTWvE))

### 🔹 Option 2: Run From Source Code

git clone <repository-link>
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

Android Setup

1. Open project in Android Studio
2. Connect device or emulator
3. Update BASE_URL if needed
4. Run the app

## 💭 How It Can Be Improved

- Add symptom auto-suggestions
- Add disease description and precautions
- Add probability confidence score
- Add user history tracking
- Improve UI/UX design
- Add authentication system

## 📈 Future Scope

- Add chatbot-style symptom input
- Add voice-based symptom detection
- Convert into a telemedicine assistant
- Deploy as full-stack web app
