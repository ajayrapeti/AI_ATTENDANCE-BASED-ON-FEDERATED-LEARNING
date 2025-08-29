This project presents a scalable, secure, and efficient attendance management system that leverages Federated Learning (FL) and InsightFace for face recognition. Unlike centralized systems, raw student images never leave the device—only model updates or embeddings are shared, ensuring privacy and compliance with regulations like GDPR.

🚀 Features

🧑‍🎓 Automated Attendance – Detects students in classroom photos.

🔐 Privacy-Preserving – Raw facial data stays on devices, only embeddings are shared.

⚡ Federated Learning – Uses FedAvg algorithm to aggregate local models.

🎭 Accurate Recognition – Employs InsightFace with ArcFace loss for robust embeddings.

📊 Analytics – Provides detailed attendance reports (present, absent, unregistered).

🏫 Scalability – Works seamlessly in large institutions.

⚙️ System Architecture

Local Processing (Client Side)

Capture student images.

Detect faces using MTCNN/Haar Cascades.

Generate embeddings with InsightFace.

Train/update local model.

Federated Server (Global Model)

Aggregates updates using Federated Averaging (FedAvg).

Refines accuracy without accessing raw data.

Sends global model back to clients.

Attendance Module

Teacher uploads a class photo.

Detected faces compared with embeddings.

Marks present, absent, or unregistered students.

Exports results in structured format (JSON/CSV).

🛠️ Tech Stack

Languages: Python

Backend: Flask / FastAPI

Libraries: OpenCV, TensorFlow / PyTorch, InsightFace, MTCNN

Machine Learning: Federated Learning (FedAvg), ArcFace loss

Deployment: Local devices + Central Federated Server

📂 Dataset

~30 students, each with 100 training images.

Group classroom photos for real-world testing.

📊 Results

Detected Students Present: 17

Absent Students: 13

Unregistered Students: 2

Faces highlighted with green (recognized) or red (unknown) bounding boxes.

✅ Conclusion

This federated approach ensures:

🔐 Enhanced privacy and security

⚡ Low latency and scalable performance

🎯 High accuracy with InsightFace embeddings

🏫 Practical deployment in educational institution
