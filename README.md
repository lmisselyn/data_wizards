# Job Description Classifier

-   This application allows to classify job descriptions that are fraudulent.

## Overview

-   The classifier is a simple Naive Bayes classifier.
-   The dataset contained approximativeley 18000 job descriptions. Around 800 of them are fake.
-   This dockerized application train the classifier on this dataset, and oprovide an API endpoint to use the model on new data.

## Installation

-   Open a terminal at the root of the project and execute the following commands

```bash
sudo docker build -t nlpclassifier .
sudo docker run -d -p 5000:5000 nlpclassifier
```

## Use of the app

-   Use the API to provide a job description to classify

-   POST request

-   Address: (http://127.0.0.1:5000/classify)

-   Body :
    `{
"text":"Job description"
}`

-   Change the port if needed

## Examples

-   "Job Functions* Data Entry for updating company information systems and databases* Reconcile weekly production reports &amp voided powers report* File Production reports, voided powers, &amp exonerations* Order Office Depot supplies* Cover Receptionist (front desk) when needed* Process outgoing mailRequirements* Knowledge in Microsoft Excel, Word and Outlook* Strong Alpha-numeric Data Entry* Attention to detail and accuracy* Ability to work under pressure to meet deadlines\* Excellent team playerStarting pay: $25/HRBenefits:• Full medical and dental benefits, additional voluntary benefits• 401K with matching"

-   Expected result : "The job description is fraudulent"

-   "Job Title: SAS Grid DeveloperDuration: 06+ months contract with potential extensionLocation: Berkeley Heights, NJ"

-   Expected result : "The job description is safe"
