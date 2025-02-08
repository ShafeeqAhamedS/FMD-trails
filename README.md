# Project Overview

The project, titled FMD: Forge Model Deploy, focuses on automating the deployment of machine learning (ML) and deep learning (DL) models. The aim is to create a seamless, user-friendly platform for developers to deploy their ML projects without manually writing backend APIs, frontend code, or configuring infrastructure. Inspired by platforms like Vercel and Heroku, this project targets ML developers who can upload or link their code repositories, leaving the deployment logic to the platform.

FMD: Forge Model Deploy is an innovative platform designed to automate the deployment of machine learning (ML) models. By addressing key pain points like manual coding, infrastructure setup, and scalability issues, FMD aims to empower developers to deploy their models effortlessly with minimal technical overhead.

## Core Components and Features
1. **Upload Files**
    - Users directly upload project files.
    - Required files: Saved Model File, Full training IPYNB File, Data sample (few rows, images, etc.)
    - A requirements.txt file is mandatory to ensure dependencies are automatically installed.
2. **Code Parsing**
    - An AI-based bot parser identifies critical components in the code (e.g., prediction logic, input requirements, and preprocessing pipelines).
    - Validation ensures that the project includes all necessary elements like prediction functions, input/output mappings, and data handling logic.
3. **Backend API Generation**
    - The platform generates FastAPI routes like /predict, /metrics, etc., based on identified logic in the code.
    - API routes are designed to work seamlessly with the parsed pipelines.
4. **Dynamic Frontend Creation**
    - A React-based or HTML form is dynamically generated to match the input fields and requirements of the ML model.
    - Users interact with this frontend to provide inputs and retrieve outputs.
    - Inputs will be pre-processed in the backend like label encoding.
5. **Cloud Deployment**
    - Applications are deployed to cloud services (AWS EC2 instances) using Jenkins pipeline.
    - Scalability is built into the architecture.
6. **Error Handling**
    - Code validation stops deployment in case of errors or missing elements in the user’s submission.
    - Logs and feedback mechanisms are provided to guide users in resolving issues.
7. **Strict Quotas and Limits**
    - Cloud storage and compute usage are monitored and controlled via strict quotas, avoiding resource overuse.

## Planned Functionalities for Future Expansion
- **Model Versioning**
    - Allowing users to deploy and manage multiple versions of the same model.
- **Microservices Architecture**
    - Each route (e.g., /predict, /metrics) can be deployed as an independent microservice for scalability.
- **CI/CD Integration**
    - Automatic deployments triggered by updates in the GitHub repository.
- **Advanced API Routes**
    - Adding routes for training (/train), metrics viewing (/metrics), health monitoring (/health), and more.

## Challenges and Solutions
1. **Code Parsing Challenges**
    - Users might have varied coding styles and naming conventions.
    - Solution: Rely on an AI-based parsing tool for advanced flexibility that looks for common practices (e.g., X, Y, predict).
2. **Handling Errors in Code**
    - Missing or incomplete pipelines could lead to deployment failures.
    - Solution: Implement validation checks before deployment and provide user feedback with error logs.
3. **Resource Constraints**
    - Users might exceed storage or compute limits.
    - Solution: Enforce strict quotas for training data and computational resources.
4. **Scalability**
    - Initial deployments might struggle with multiple requests or models.
    - Solution: Future-proof architecture with options for scaling routes as microservices.

## Tools and Technologies Used
- **Backend**: FastAPI for route generation and API logic.
- **Frontend**: React for dynamic input forms; HTML as a fallback for simplicity.
- **Cloud Services**: AWS EC2 for deployment; AWS metrics for monitoring.
- **Integration**: GitHub for repository linking; requirements.txt for dependency management.
- **Languages**: Python for backend logic, JavaScript for frontend React forms.

## Key Objectives
1. **Automate the Deployment Process**
    - Eliminate the need for developers to write custom backend APIs or frontend interfaces by automating these processes.
2. **User-Friendly Experience**
    - Provide an intuitive platform where users can simply upload their projects or link their GitHub repositories.
3. **Flexible and Scalable Architecture**
    - Build a system that supports scaling and integrates with modern cloud infrastructure, with future plans for microservices and advanced API routes.
4. **Seamless Integration and Feedback**
    - Ensure seamless integration with existing developer workflows and provide actionable feedback in case of issues.

## User Journey
1. **Step 1: Uploading Project**
    - Users can upload files directly.
    - Mandatory: requirements.txt to manage dependencies.
2. **Step 2: Parsing the Code**
    - An AI bot parser identifies:
        - Model Pipelines: Prediction, preprocessing, and training logic.
        - Inputs and Outputs: Defined fields required for user interaction.
        - Dependencies: Libraries required for execution.
    - Validation ensures the project contains the minimum necessary components.
3. **Step 3: Generating Backend and Frontend**
    - Backend: A FastAPI-based API is created with predefined routes such as /predict for inference.
    - Frontend: A React-based or simple HTML form dynamically generates fields based on the model’s input requirements.
4. **Step 4: Deployment**
    - The system deploys the generated backend and frontend to AWS (Lambda or EC2).
    - Resources are allocated based on predefined quotas.
5. **Step 5: Interaction and Monitoring**
    - Users access the frontend to interact with the model.
    - AWS metrics provide insights into usage, errors, and performance.

## ENV File
```python
API_KEY = "API_KEY"
MODEL_NAME = "MODEL_NAME" # Ex: gemini-2.0-flash
```

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Place needed files in User_Input folder
```
IPYNB File 
DATASET - csv
Saved Model File
```

## Run
```bash
python main.py
```