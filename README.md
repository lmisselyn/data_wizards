# Job Description Classifier

-   This application allows to classify job descriptions that are fraudulent.

## Overview

-   The classifier is a simple Naive Bayes classifier.
-   The dataset contained approximativeley 18000 job descriptions. Around 800 of them are fake.
-   This dockerized application train the classifier on this dataset, and oprovide an API endpoint to use the model on new data.

## Installation

-   open a terminal at the root of the project and execute the following commands

```bash
sudo docker build -t nlpclassifier .
sudo docker run -d -p 5000:5000 nlpclassifier
```

## Use of the app

-   Use the API to provide a job description to classify

-   Address: (http://localhost:5000)

-   Body :
    '{
    "text":"Job description"
    }'
