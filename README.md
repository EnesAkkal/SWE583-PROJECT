## Prerequisites
- Python 3.13.0

---

## Folder Structure
```
project/
├── data/                        # Contains dataset and frames
│   ├── frames/                  # Processed video frames
│   └── UCF-101/                 # Original dataset
├── results/                     # Stores model training results
│   ├── classes_100_*            # Results for CNN Model
│   ├── vit_classes_*            # Results for ViT model
│   └── selected_classes.txt     # List of selected action classes
├── venv/                        # Virtual environment (optional)
├── main.py                      # Script specifically for CNN training
├── vit_main.py                  # Script specifically for ViT training
├── requirements.txt             # Dependencies for the project
└── README.md                    # Documentation
```

---


## Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/EnesAkkal/SWE583-PROJECT.git
   cd project
   ```

2. **Create a virtual environment **  
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### **Train the Models**
```bash
python vit_main.py
python cnn_main.py
```
- The script will prompt whether to reuse previously selected classes.  

### **Metrics and Results**  
- Training and validation metrics such as loss and accuracy are logged and displayed after each epoch.  
- Results are saved under the `results/` folder for further analysis.

---