Loan Eligibility predictions for Dream housing finanace company
---

Problem Statement
---
Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas.

Customer first applies for home loan and after that company validates the customer eligibility for loan.

Company wants to automate the loan eligibility process(real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, number of dependents, income loan amount credit history and others. To automate this process, they provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers.

Dataset
---
This dataset is downloaded from analytic vidya's hackthon.

You can find the dataset [here](https://github.com/madhuri-15/mlbookcamp-homeworks/tree/main/Mid-term%20Project/data)

There are two csv data files.
* [train.csv](https://github.com/madhuri-15/mlbookcamp-homeworks/tree/main/Mid-term%20Project/data/train.csv) : This dataset can be used for exploratary data analsis and for training machine learning model.
* [test.csv](https://github.com/madhuri-15/mlbookcamp-homeworks/tree/main/Mid-term%20Project/data/test.csv) : This dataset can be used to predict the output using ML model.

|variables | description|
|-|-|
|loan id | unqiue loan id|
|gender|  male/female|
|married | applicant married(Y/N)|
|dependents | number of dependents on applicant|
|education | applicant education(graduate/under graduate)|
|self-employed | self employed (y/n)|
|applicant-income | applicant income|
|co-applicant income | coapplicant who is apply with applicant as money borrowers.|
|loan amount | loan amount in thousands|
|loan amount term | term of loan in months - |amount of time the lender gives you to repay your laon.|
|credit history | credit history meets guidlines|
|property area | urban/semi urban / rural|
|loan status | loan approved(y/n)|
