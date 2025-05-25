# RAG_Workflow
This repository demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using **Milvus** as a vector database. It includes code, notebooks, and configuration files to help you set up and run a basic RAG workflow.

---

## 📁 Project Structure
```
RAG_Workflow/
├── research_notebook/       # Jupyter Notebooks for experimentation and testing
├── src/                     # Python source code for data processing and retrieval
├── volumes/milvus/          # Data volume and configuration for Milvus
├── embedEtcd.yaml           # Milvus embedding configuration
├── standalone_embed.sh      # Script to launch Milvus in standalone mode
├── user.yaml                # User configuration for Milvus
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT License
├── README.md                # Project documentation


---

## ⚙️ Requirements

- Python 3.x
- Docker & Docker Compose
- Milvus
- Jupyter Notebook / Lab

---

## 🚀 Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MahaSaad12/RAG_Workflow.git
   cd RAG_Workflow

-----

###
2 **Start Milvus (Standalone Mode)**
bash standalone_embed.sh

###
3. **Open Notebook***
   Launch Jupyter Lab/Notebook and navigate to the research_notebook/ folder.
###   
4. **Use the source code***
All main code components (e.g., ingestion, retrieval) are located in the src/ folder.



**Author**
Maha Saad
