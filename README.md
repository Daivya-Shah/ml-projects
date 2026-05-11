# Supervised Machine Learning Projects

This repo contains two end-to-end machine learning projects built with scikit-learn. Both use Support Vector Machines (SVM) for binary classification and follow the same general workflow: data preprocessing, standardization, train/test split, model training, and a predictive system you can feed new values into.

---

## Project 1: Crane Maintenance Prediction

Cranes are expensive to repair and dangerous when they fail unexpectedly. This project builds a classifier that predicts whether a crane needs maintenance based on its operational and environmental data.

### Dataset

The dataset has around 25,000 rows covering 10 different cranes. The target label (`Maintenance_Required`) is engineered from domain knowledge using these rules:

| Condition | Threshold |
|---|---|
| Motor current | > 375 Amps |
| Hydraulic pressure + low oil | > 2250 psi AND oil level < 25% |
| Crane age | > 15 years |
| Past failures | > 7 |
| Daily usage with low maintenance | usage = 7 days/week AND < 4 services/year |

### Features

| Feature | Description | Range |
|---|---|---|
| Crane_ID | Identifier for each crane | 1-10 |
| Ambient_Temperature | Temperature around the crane | 0-50°C |
| Humidity | Environmental humidity | 30-90% |
| Wind_Speed | Wind speed at crane location | 0-20 m/s |
| Operation_Type | 0 = Loading, 1 = Unloading, 2 = Moving | 0-2 |
| Operation_Hours | Hours operational per day | 0-24 hrs |
| Load_Weight | Weight handled | 1-50 tons |
| Number_of_Lifts | Lifts performed | 1-300 |
| Motor_Current | Current drawn by motor | 10-500 Amps |
| Hydraulic_Pressure | Hydraulic system pressure | 500-3000 psi |
| Oil_Level | Oil level percentage | 0-100% |
| Oil_Viscosity | Hydraulic oil viscosity | 10-1000 cSt |
| Average_Daily_Motor_RPM | Motor RPM | 0-10,000 RPM |
| Peak_Load_Weight | Maximum load handled | 1-50 tons |
| Time_Since_Last_Maintenance | Days since last service | 0-365 days |
| Crane_Age | Age of the crane | 0-30 years |
| Usage_Frequency | Days per week in use | 1-7 days |
| Number_of_Past_Failures | Historical failure count | 0-9 |
| Maintenance_Frequency | Services per year | 1-12 |

### Pipeline

1. Standardize features using `StandardScaler`
2. Stratified 80/20 train/test split
3. Train a linear SVM classifier
4. Evaluate on both splits
5. Run predictions on new input data

### Results

| Split | Accuracy |
|---|---|
| Training | 83.93% |
| Testing | 84.66% |

---

## Project 2: Diabetes Prediction

This project predicts whether a patient is diabetic based on health metrics from the Pima Indians Diabetes Dataset.

### Dataset

768 patients, 8 features, binary outcome (0 = non-diabetic, 1 = diabetic). The dataset is moderately imbalanced with 500 non-diabetic and 268 diabetic cases.

From exploratory analysis, diabetic patients tend to have higher glucose levels, higher BMI, and are on average older than non-diabetic patients.

### Features

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Genetic diabetes risk score |
| Age | Age in years |

### Pipeline

1. Standardize features using `StandardScaler`
2. Stratified 80/20 train/test split
3. Train a linear SVM classifier
4. Evaluate on both splits
5. Run predictions on new input data

### Results

| Split | Accuracy |
|---|---|
| Training | 78.66% |
| Testing | 77.27% |

---

## Tech Stack

- Python 3
- NumPy
- pandas
- scikit-learn
