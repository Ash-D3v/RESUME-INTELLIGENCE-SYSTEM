# Resume Intelligence System

A comprehensive AI-powered full-stack system for resume analysis, job matching, skill assessment, and career recommendations with a modern React frontend and FastAPI backend.

## Features

- **Resume Parsing**: Extract and analyze resume content using advanced NLP techniques
- **ATS Scoring**: Evaluate resumes against Applicant Tracking Systems compatibility
- **Job Matching**: Intelligent matching of resumes to job descriptions
- **Skill Analysis**: Identify and assess key skills from resumes
- **Course Recommendations**: Suggest relevant courses based on skill gaps
- **ML-Enhanced Analysis**: Machine learning models for better insights

## Tech Stack

### Backend
- **Language**: Python 3.11+
- **Web Framework**: FastAPI
- **Machine Learning**: scikit-learn, NLTK, spaCy
- **Data Processing**: pandas, numpy
- **Deployment**: Docker, Docker Compose

### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **UI Components**: Radix UI + Tailwind CSS
- **State Management**: TanStack Query
- **Routing**: React Router
- **Icons**: Lucide React

## Project Structure

```
Backend/
├── backend/                # FastAPI Backend
│   ├── main.py            # Main FastAPI application
│   ├── resume_parser.py   # Resume parsing functionality
│   ├── ats_scorer.py      # ATS compatibility scoring
│   ├── job_matcher.py     # Job-resume matching algorithm
│   ├── skill_analyzer.py  # Skill extraction and analysis
│   ├── course_recommender.py # Course recommendation engine
│   ├── ml_enhanced_analyzer.py # ML-based analysis
│   ├── ml_models/         # Trained ML models
│   ├── requirements.txt   # Python dependencies
│   ├── Dockerfile         # Docker configuration
│   └── docker-compose.yml # Multi-container setup
├── frontend/              # React Frontend
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── pages/         # Page components
│   │   ├── lib/           # API service layer
│   │   └── App.tsx        # Main app component
│   ├── package.json       # Node.js dependencies
│   ├── vite.config.ts     # Vite configuration
│   └── .env               # Environment variables
└── README.md              # This file
```

## Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm or yarn

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:8080`

### Full Stack Development

1. **Terminal 1** - Start Backend:
```bash
cd backend
python main.py
```

2. **Terminal 2** - Start Frontend:
```bash
cd frontend
npm run dev
```

3. Open your browser to `http://localhost:8080`

### Using Docker (Backend Only)

1. Build and run with Docker Compose:
```bash
cd backend
docker-compose up --build
```

Or use the provided script:
```bash
./run-docker.sh
```

## API Endpoints

### Core Endpoints
- `POST /upload-resume` - Upload and analyze resume file
- `POST /score` - Score resume against job description
- `GET /jobs?skills=` - Get job suggestions based on skills
- `GET /skill-gap?userSkills=&roleRequiredSkills=` - Analyze skill gaps
- `GET /courses?skills=` - Get course recommendations
- `GET /analytics` - Get comprehensive analytics
- `POST /detailed-analysis` - Get detailed resume analysis

### ML-Enhanced Endpoints
- `POST /ml-enhanced-matching` - ML-powered skill matching
- `POST /predict-job-fit` - Predict job fit probability
- `GET /ml-job-recommendations` - ML-based job recommendations
- `POST /train-ml-models` - Train ML models
- `GET /ml-analytics` - ML model performance analytics

### Utility Endpoints
- `GET /health` - Health check
- `DELETE /clear-data` - Clear stored data

For detailed API documentation, see [COMPREHENSIVE_API_DOCUMENTATION.md](backend/COMPREHENSIVE_API_DOCUMENTATION.md)

## Frontend-Backend Integration

The frontend connects to the backend through:

1. **API Service Layer**: `/frontend/src/lib/api.ts` - Centralized API calls
2. **Configuration**: `/frontend/src/lib/api-config.ts` - API base URL configuration
3. **Environment Variables**: `/frontend/.env` - Backend URL configuration

### Key Integration Points:

- **Dashboard**: Real-time resume analysis with file upload
- **Jobs Page**: Dynamic job matching based on user skills
- **Analytics**: Live data from backend analytics endpoints
- **ML Tools**: Integration with ML-enhanced features

## Docker Deployment

For production deployment, refer to [DOCKER_README.md](backend/DOCKER_README.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
