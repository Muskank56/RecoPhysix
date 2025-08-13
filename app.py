import streamlit as st
import pandas as pd
import joblib
import random
import os
import time

# --- Load Model and Encoders ---
MODEL_PATH = "quiz_recommender_model.joblib"
LE_DIFF_PATH = "label_encoder_diff.joblib"
LE_NEXT_PATH = "label_encoder_next.joblib"

try:
    model = joblib.load(MODEL_PATH)
    le_diff = joblib.load(LE_DIFF_PATH)
    le_next = joblib.load(LE_NEXT_PATH)
except Exception as e:
    st.error(f"Error loading model/encoders: {e}")
    st.stop()

# --- Quiz Data Folder ---
QUIZ_FOLDER = "Quizzes"
if not os.path.exists(QUIZ_FOLDER):
    st.error(f"Quiz folder '{QUIZ_FOLDER}' not found!")
    st.stop()

# --- UI Title ---
st.title("RecoPhysix – Personalized Physics Quiz Recommender")

# --- Exam Selection ---
exam_choice = st.selectbox("Select your exam:", ["NEET", "JEE"])

# --- Topic Selection ---
quiz_files = [f for f in os.listdir(QUIZ_FOLDER) if f.endswith(".csv")]
topics = [os.path.splitext(f)[0] for f in quiz_files]
selected_topic = st.selectbox("Choose a quiz topic:", topics)

# --- Initialize Session State ---
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None

# --- Start Quiz Button ---
if st.button("Start Quiz"):
    st.session_state.quiz_started = True
    st.session_state.start_time = time.time()
    st.success("Quiz started! Timer is running...")

# --- Load Selected Quiz ---
quiz_path = os.path.join(QUIZ_FOLDER, f"{selected_topic}.csv")
quiz_df = pd.read_csv(quiz_path)

def normalize_text(s):
    if s is None:
        return ""
    return str(s).strip().lower()

# Limit to 30 questions max
quiz_df = quiz_df.sample(min(30, len(quiz_df)), random_state=42).reset_index(drop=True)

# --- Show Quiz Only If Started ---
if st.session_state.quiz_started:

    st.subheader(f"Quiz: {selected_topic}")

    # Display questions
    for i, row in quiz_df.iterrows():
        q_key = f"q_{i}"
        q_text = row.get('Question') or row.get('question') or f"Question {i+1}"
        opts = [
            str(row.get('option_1') or row.get('Option1') or ""),
            str(row.get('option_2') or row.get('Option2') or ""),
            str(row.get('option_3') or row.get('Option3') or ""),
            str(row.get('option_4') or row.get('Option4') or ""),
        ]
        if q_key not in st.session_state:
            st.session_state[q_key] = None
        st.write(f"**Q{i+1}. {q_text}**")
        st.radio("", opts, key=q_key)

    # --- Submit Quiz Button ---
    if st.button("Submit Quiz", key="submit_quiz"):
        # Ensure quiz was started
        if st.session_state.start_time is None:
            st.warning("Please click 'Start Quiz' first.")
        else:
            # Get selected answers as option numbers
            selected_indices = []
            for i, row in quiz_df.iterrows():
                sel_text = st.session_state.get(f"q_{i}")
                if sel_text is None:
                    selected_indices.append(None)
                    continue
                opts = [
                    str(row.get('option_1') or row.get('Option1') or ""),
                    str(row.get('option_2') or row.get('Option2') or ""),
                    str(row.get('option_3') or row.get('Option3') or ""),
                    str(row.get('option_4') or row.get('Option4') or ""),
                ]
                found_idx = None
                for j, opt_text in enumerate(opts, start=1):
                    if normalize_text(opt_text) == normalize_text(sel_text):
                        found_idx = j
                        break
                selected_indices.append(found_idx)

            # Detect correct answer column
            corr_col = None
            for c in quiz_df.columns:
                if 'correct' in c.lower():
                    corr_col = c
                    break

            correct_indices = []
            for i, row in quiz_df.iterrows():
                if corr_col is None:
                    correct_indices.append(None)
                    continue
                val = row[corr_col]
                if pd.isna(val):
                    correct_indices.append(None)
                    continue
                try:
                    idx = int(float(val))
                    if 1 <= idx <= 4:
                        correct_indices.append(idx)
                        continue
                except Exception:
                    pass
                opts = [
                    str(row.get('option_1') or row.get('Option1') or ""),
                    str(row.get('option_2') or row.get('Option2') or ""),
                    str(row.get('option_3') or row.get('Option3') or ""),
                    str(row.get('option_4') or row.get('Option4') or ""),
                ]
                found_idx = None
                for j, opt_text in enumerate(opts, start=1):
                    if normalize_text(opt_text) == normalize_text(val):
                        found_idx = j
                        break
                correct_indices.append(found_idx)

            # Calculate score
            correct_count = sum(
                sel is not None and corr is not None and sel == corr
                for sel, corr in zip(selected_indices, correct_indices)
            )
            total_questions = len(correct_indices)
            total_answered = sum(sel is not None for sel in selected_indices)

            # Scale score to 100 marks
            score = (correct_count / total_questions) * 100
            st.success(f"Your score: {score:.2f} / 100  (Answered: {total_answered})")

            # Time taken
            end_time = time.time()
            time_taken_value = round(end_time - st.session_state.start_time, 2)
            st.info(f"Time taken: {time_taken_value} seconds")

            # Difficulty encoding
            percent_score = score
            if percent_score < 40:
                difficulty = "Easy"
            elif percent_score < 70:
                difficulty = "Medium"
            else:
                difficulty = "Hard"
            diff_enc_value = le_diff.transform([difficulty])[0]

            quiz_attempt_value = 1  # static for now

            # Predict next quiz
            input_data = pd.DataFrame([{
                "diff_enc": diff_enc_value,
                "score": percent_score,
                "quiz_attempt": quiz_attempt_value,
                "time_taken": time_taken_value
            }])
            input_data = input_data[model.feature_names_in_]

            try:
                prediction_enc = model.predict(input_data)[0]
                predicted_topic = le_next.inverse_transform([prediction_enc])[0]

                # --- Message logic ---
                if score < 40:
                    message = f"😐 You need more practice. Let's go back to an easier topic: {predicted_topic}."
                elif score <= 60:
                    message = f"🙂 You're doing okay. Let's try another medium topic: {predicted_topic}."
                elif score <= 80:
                    message = f"🔥 Great job! You're ready for a harder topic: {predicted_topic}."
                else:
                    message = f"🎉 Well done! You're ready for a mixed questions practice: {predicted_topic}."

                st.success(message)

            except Exception as e:
                st.error(f"Error making recommendation: {e}")
