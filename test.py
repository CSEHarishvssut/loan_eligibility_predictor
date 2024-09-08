# Standard library imports
import warnings # For warning handling

# Third-party imports
import pandas as pd # for data manipulation and analysis, CSV file I/O
import numpy as np # For numerical operations and mathematical functions
import matplotlib.pyplot as plt # For data visualization
import seaborn as sns # For statistical graphics
from sklearn.model_selection import train_test_split # For data splitting (Training & Testing) in machine learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler # For feature standardization & Normalization
from sklearn.metrics import accuracy_score, classification_report # For model evaluation
from termcolor import colored # For colored text printing

# For warning handling
warnings.filterwarnings('ignore') # For ignoring warnings

# Print a success message in green color
print(colored("THE REQUIRED LIBRARIES WERE SUCCESSFULLY IMPORTED...", "green", attrs=['reverse']))

# Define file path
# The "r" in front of the string makes it a "raw string".  It tells Python to treat backslashes (\) in the string as normal characters, not special ones.
file_path = r'/kaggle/input/finance-loan-approval-prediction-data/train.csv'

try:
    # Read the CSV file and save it in "loan_data" variable
    loan_data = pd.read_csv(file_path)
    
    # Print a success message
    print(colored("THE DATASET LOADED SUCCESSFULLY...", "green", attrs=['reverse']))

except FileNotFoundError:
    # Handle file not found error
    print(colored("ERROR: File not found!", "red", attrs=['reverse']))

except Exception as e:
    # Handle other exceptions
    print(colored(f"ERROR: {e}", "red", attrs=['reverse']))



# Displaying the first 7 rows.
loan_data_rows = loan_data.head(7)  # .head() the default value = 5

print(colored('As you can see, the first 7 rows in the dataset:\n', 'green', attrs=['reverse']))

# Iterate over each row in the loan_data_rows DataFrame
for index, row in loan_data_rows.iterrows():
    # Print the index label of the current row, "index + 1" start with 1 not 0 
    print(colored(f"Row {index + 1}:", "white", attrs=['reverse']))
    
    # Print the content of the current row
    print(row)
    
    # Print a separator line
    print("--------------------------------------")


# .iterrows() function:
# Returns: (index, data|row|series)
#     Index: the index of the row.
#     Data: the data of the row as a series.
# Note: The column names will also be returned, in addition to the specified rows.


# Show the shape of the dataset
print("The shape =", loan_data.shape)

# Dataset dimensions and statistics
num_rows, num_cols = loan_data.shape
num_features = num_cols - 1
num_data = num_rows * num_cols

# Print the information about the dataset
print(f"Number of Rows: {num_rows}")
print(f"Number of Columns: {num_cols}")
print(f"Number of Features: {num_features}")
print(f"Number of All Data: {num_data}")

# Check and ensure running
print(colored("The task has been completed without any errors....", "green", attrs=['reverse']))


# Get basic information from dataset like "Column names", "Data types" and "Non-null values counts"
print(loan_data.info())

# Check and ensure running
print(colored("The task has been completed without any errors....", "green", attrs=['reverse']))


# Descriptive Statistics of Numeric Variables only.
loan_data.describe().T.round(2)

# Descriptive analysis of Categorical Variables only.
loan_data.describe(include=object)

# "dropna" as False to count NaN (Not-a-Number) values
GenderAnalysis = loan_data.Gender.value_counts(dropna=False)
print(GenderAnalysis)

# Bar Charts Analysis "For Gender feature"
sns.countplot(x="Gender", data=loan_data, palette="flare")
plt.show()

# "dropna" as False to count NaN values
MarriedAnalysis = loan_data.Married.value_counts(dropna=False)
print(MarriedAnalysis)

# Create a pie chart "For Married feature"
plt.figure(figsize=(10, 5)) # figure in inches

# labels by descending order
plt.pie(MarriedAnalysis, 
        labels=[("Married"),("Single"),("NaN")], 
        startangle=216, 
        autopct='%1.1f%%', 
        colors=sns.color_palette("flare", 
        len(MarriedAnalysis)))

plt.axis('equal')  # Used to set the aspect ratio of the plot to be equal.
plt.title('Marital Status Distribution')
plt.show()

# "dropna" as False to count NaN values
DependentsAnalysis = loan_data.Dependents.value_counts(dropna=False)
print(DependentsAnalysis)

# Bar Charts Analysis "For Dependents feature"
sns.countplot(x="Dependents", data=loan_data, palette="flare")
plt.show()

# "dropna" as False to count NaN values
EducationAnalysis = loan_data.Education.value_counts(dropna=False)
print(EducationAnalysis)

# Bar Charts Analysis "For Education feature"
sns.countplot(x="Education", data=loan_data, palette="flare")
plt.show()

# "dropna" as False to count NaN values
Self_EmployedAnalysis = loan_data.Self_Employed.value_counts(dropna=False)
print(Self_EmployedAnalysis)

# Bar Charts Analysis "For Self Employed feature"
sns.countplot(x="Self_Employed", data=loan_data, palette="flare")
plt.show()

# Calculate the average income
average_income = loan_data['ApplicantIncome'].mean()
print(f"The Average Income: {average_income:.2f} ")

# Count incomes higher and lower than average
above_average_count = (loan_data['ApplicantIncome'] > average_income).sum()
below_average_count = (loan_data['ApplicantIncome'] <= average_income).sum()

# Calculate ratio and print the results
ratio = above_average_count / below_average_count
print(f"The ratio of people with income above average to below average: {ratio*100:.2f} ")
print(f"Number of people income above the average: {above_average_count}")
print(f"Number of people income below the average: {below_average_count}")

# Plot the ratio using Seaborn
plt.figure(figsize=(8, 4))
sns.barplot(x=['Above Average', 'Below Average'], y=[above_average_count, below_average_count], palette="flare")
plt.title('Ratio of People with Income Above Average to Below Average')
plt.ylabel('Count')
plt.show()

# "dropna" as False to count NaN values
Credit_HistoryAnalysis = loan_data.Credit_History.value_counts(dropna=False)
print(Credit_HistoryAnalysis)

# Bar Charts Analysis "For Credit History feature"
sns.countplot(x="Credit_History", data=loan_data, palette="flare")
plt.show()

# "dropna" as False to count NaN values
Property_AreaAnalysis = loan_data.Property_Area.value_counts(dropna=False)
print(Property_AreaAnalysis)

# Bar Charts Analysis "For Property Area feature"
sns.countplot(x="Property_Area", data=loan_data, palette="flare")
plt.show()

# "dropna" as False to count NaN values
Loan_StatusAnalysis = loan_data.Loan_Status.value_counts(dropna=False)
print(Loan_StatusAnalysis)

# Bar Charts Analysis "For Loan Status column"
sns.countplot(x="Loan_Status", data=loan_data, palette="flare")
plt.show()

# "dropna" as False to count NaN values
Loan_Amount_TermAnalysis = loan_data.Loan_Amount_Term.value_counts(dropna=False)
print(Loan_Amount_TermAnalysis)

# Bar Charts Analysis "For Loan amount term feature"
sns.countplot(x="Loan_Amount_Term", data=loan_data, palette="flare")
plt.show()

