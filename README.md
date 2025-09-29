# Resume Intelligence System

A comprehensive AI-powered backend system for resume analysis, job matching, skill assessment, and career recommendations.

## Features

- **Resume Parsing**: Extract and analyze resume content using advanced NLP techniques
- **ATS Scoring**: Evaluate resumes against Applicant Tracking Systems compatibility
- **Job Matching**: Intelligent matching of resumes to job descriptions
- **Skill Analysis**: Identify and assess key skills from resumes
- **Course Recommendations**: Suggest relevant courses based on skill gaps
- **ML-Enhanced Analysis**: Machine learning models for better insights

## Tech Stack

- **Language**: Python 3.11+
- **Web Framework**: FastAPI
- **Machine Learning**: scikit-learn, NLTK, spaCy
- **Data Processing**: pandas, numpy
- **Deployment**: Docker, Docker Compose

## Project Structure

```
backend/
├── main.py                 # Main FastAPI application
├── resume_parser.py        # Resume parsing functionality
├── ats_scorer.py          # ATS compatibility scoring
├── job_matcher.py         # Job-resume matching algorithm
├── skill_analyzer.py      # Skill extraction and analysis
├── course_recommender.py  # Course recommendation engine
├── ml_enhanced_analyzer.py # ML-based analysis
├── ml_models/             # Trained ML models
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Multi-container setup
└── run-docker.sh         # Docker startup script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ash-D3v/RESUME-INTELLIGENCE-SYSTEM.git
cd RESUME-INTELLIGENCE-SYSTEM
```

2. Navigate to backend directory:
```bash
cd backend
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Locally

1. Start the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Using Docker

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

Or use the provided script:
```bash
./run-docker.sh
```

## API Endpoints

- `POST /analyze-resume` - Analyze a resume file
- `POST /match-jobs` - Match resume to job descriptions
- `GET /skill-analysis` - Get skill analysis results
- `GET /course-recommendations` - Get course recommendations

For detailed API documentation, see [COMPREHENSIVE_API_DOCUMENTATION.md](backend/COMPREHENSIVE_API_DOCUMENTATION.md)

## Docker Deployment

For production deployment, refer to [DOCKER_README.md](backend/DOCKER_README.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
