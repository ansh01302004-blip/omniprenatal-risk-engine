# 🧬 OmniPrenatal Risk Engine

### 👨‍💻 **Project Architect:** Ansh Jadav (Student ID: 125119712)  
### 🏥 **Classification:** Proprietary Clinical Decision Support System (CDSS)

---

## 📋 Project Overview
The **OmniPrenatal Risk Engine** is a live, end-to-end predictive clinical middleware application designed to eliminate diagnostic ambiguity and latency in maternal-fetal medicine. By integrating fragmented data streams—such as maternal clinical conditions, paternal toxicological exposures, co-carrier genetic screen statuses, and family lineage history—the engine processes these vectors through an optimized machine learning pipeline to deliver precise, real-time risk evaluations.

---

## 🛠️ The 4-Pillar Architecture Pipeline

1. **Acquisition Layer:** Streamlines the continuous ingestion of multi-source patient intake metrics (demographics, toxicology, genomic screenings).
2. **Integration Layer:** Securely maps disparate raw data vectors to a unified relational framework, writing clean transaction assets directly to data storage (`parental_clinical_data.xlsx`).
3. **Analysis Layer:** Deploys an optimized **Random Forest Classifier** ensemble (comprising 100 independent decision trees) to decode complex feature intersections.
4. **Delivery Layer:** Serves calculated insights through a high-fidelity web interface powered by **Streamlit**, routing high-risk profiles instantly to laboratory supervisors.

---

## 📊 Core Technology Stack
* **Language:** Python 3.x
* **Framework:** Streamlit (User Interface & Interactive Controls)
* **Machine Learning:** Scikit-Learn (Random Forest Ensemble)
* **Data Processing:** Pandas, NumPy, OpenPyXL (Excel Integration)

---

## 🔐 Compliance & Deployment
This framework is built to adapt seamlessly to open-cloud deployment ecosystems. It has been successfully compiled and served publicly using **Hugging Face Spaces** to demonstrate containerized clinical informatics pipelines in action.

*Designed exclusively for academic evaluation and clinical conceptual modeling.*
