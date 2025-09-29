# Docker Setup for Resume Analyzer Backend

Simple Docker setup to run the Resume Analysis & Job Matching backend API.

## 🚀 Quick Start

### 1. Start the Backend
```bash
./run-docker.sh
```

### 2. Stop the Backend
```bash
./stop-docker.sh
```

## 🌐 Access Points

Once running, you can access:

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🔧 Manual Commands

### Start
```bash
docker-compose up --build -d
```

### Stop
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f
```

### Restart
```bash
docker-compose restart
```

## 📁 Files

- **`Dockerfile`** - Container configuration
- **`docker-compose.yml`** - Service orchestration
- **`run-docker.sh`** - Start script
- **`stop-docker.sh`** - Stop script

## 📊 API Endpoints

- `POST /upload-resume` - Upload and analyze resumes
- `POST /score` - ATS scoring
- `GET /jobs?skills=` - Job recommendations
- `GET /skill-gap` - Skill gap analysis
- `GET /courses?skills=` - Course recommendations

That's it! Simple and focused. 🎯
