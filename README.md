# **Financial Risk Detection**  
**Course Code**: CSL2010  
**Mentor**: Dr. Avinash Sharma  

---

## **Description**  
This project aims to build a machine learning-based **Financial Risk Assessment System** to classify individuals or entities into different risk categories (e.g., **low**, **medium**, **high**).  

The workflow includes:  
- Extensive **Exploratory Data Analysis (EDA)** to uncover key patterns and understand the dataset.  
- **Preprocessing** steps to clean and transform the data for optimal model performance.  

### **Key Highlights**  
- Utilizes machine learning algorithms such as:  
  - Logistic Regression  
  - Support Vector Classifier (SVC)  
  - Decision Trees  
  - Random Forest  
  - XGBoost  

- Provides **feature importance analysis** to identify critical risk factors, delivering actionable insights for financial institutions.  

---

## **Deployment**  

Follow the instructions below to set up and deploy the project locally:  

### **1. Set Up a Folder in VS Code**  
- Create a new folder named `Deployment` in VS Code.  
- Save the following files inside the folder:  
  - `app_final.py`  
  - `financial_risk_assessment.csv`  
  - `Project_Code_final.py`  

### **2. Run the Project Code**  
- In the `Deployment` folder, run the file `Project_Code_final.py` to set up necessary dependencies and models.

### **3. Run the Application**  
- After successfully running the project code, execute the file `app_final.py` to launch the web application.
  
### **4(a). Open the Folder on Your Computer**  
- Navigate to the folder location on your computer (outside of VS Code).
OR
### **4(b). Open the Terminal in the Folder**  
- Open the terminal in the same folder where the project files are stored.

### **5. Run the Streamlit Application**  
- Type the following command in the terminal:  
  ```bash
  streamlit run app_final.py
