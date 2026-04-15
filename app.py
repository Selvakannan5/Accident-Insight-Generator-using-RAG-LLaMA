import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from ollama import Client
from datetime import datetime

DATASET_PATH = "virtual_city_accidents_dataset.csv"
OLLAMA_MODEL = "llama3.2:latest"

@st.cache_data
def load_dataset_and_index():
    df = pd.read_csv(DATASET_PATH)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    accident_embeddings = embedder.encode(df['Accident_Report'].tolist(), convert_to_tensor=False)
    accident_embeddings = np.array(accident_embeddings)
    dimension = accident_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(accident_embeddings)
    return df, embedder, index

st.write("🔄 Loading model and dataset...")
df, embedder, index = load_dataset_and_index()
client = Client(host='http://localhost:11434')
st.success(f"✅ Model and data loaded. ({len(df)} reports)")

def generate_precaution_ollama(accident_id, date_time, location, vehicle, weather, road, cause, report, k=5, model=OLLAMA_MODEL):
    new_report = (
        f"Accident ID: {accident_id}\n"
        f"Date & Time: {date_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Location: {location}\n"
        f"Vehicle: {vehicle}\n"
        f"Weather: {weather}\n"
        f"Road Condition: {road}\n"
        f"Cause: {cause}\n"
        f"Report: {report}"
    )
    
    new_embedding = embedder.encode([new_report])[0]
    D, I = index.search(np.array([new_embedding]), k)
    similar_reports = df.iloc[I[0]]['Accident_Report'].tolist()
    context = "\n".join(f"- {r}" for r in similar_reports)
    
    prompt = f"""
You are a government safety expert analyzing traffic accident data.

Based on the following past accident reports:

{context}

And this new accident report with details:
{new_report}

Please suggest **3 to 5 short, concise bullet points** of high-impact safety regulations or preventive actions that the government or local authorities should implement to reduce such accidents. Use simple language and keep it brief. Focus only on systemic or infrastructural improvements.
Also predict the potential areas where accidents can happen(Give in points) Give unique suggestions for each accident according to the given report.
"""
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt.strip()}]
    )
    return response['message']['content']

st.title("🚧 Accident Precaution Generator")

accident_id = st.text_input("Accident ID")
date_time = st.date_input("Date of Accident", value=datetime.now())
time_input = st.time_input("Time of Accident", value=datetime.now().time())
datetime_combined = datetime.combine(date_time, time_input)

location = st.text_input("Location")
vehicle = st.text_input("Vehicle Type")
weather = st.text_input("Weather Conditions")
road = st.text_input("Road Condition")
cause = st.text_input("Cause of Accident")
report = st.text_area("Detailed Accident Report", height=150)

if st.button("Generate Safety Precautions"):
    if any(not val.strip() for val in [accident_id, location, vehicle, weather, road, cause, report]):
        st.warning("⚠️ Please fill in all fields.")
    else:
        with st.spinner("Generating safety precautions..."):
            try:
                output = generate_precaution_ollama(
                    accident_id, datetime_combined, location, vehicle, weather, road, cause, report, k=5
                )
                st.subheader("🛡️ Suggested Precautions:")
                st.markdown(output)
            except Exception as e:
                st.error(f"❌ Error: {e}")
