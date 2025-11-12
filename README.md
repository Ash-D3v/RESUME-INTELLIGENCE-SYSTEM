# 📄 Resume Intelligence System

> **AI-Powered Resume Analysis & Job Matching Platform** 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-3178C6?logo=typescript)](https://www.typescriptlang.org/)

---

## 🎯 Overview

**Resume Intelligence System** is a comprehensive full-stack platform that leverages **AI and Machine Learning** to analyze resumes, score them against job descriptions, match candidates with opportunities, and provide personalized course recommendations. It's designed for job seekers, recruiters, and career development professionals.

### ✨ Key Highlights
- 🤖 **AI-Powered Analysis**: Intelligent resume parsing and skill extraction
- 📊 **ATS Optimization**: Score resumes against Applicant Tracking Systems
- 💼 **Smart Job Matching**: Match resumes with relevant opportunities
- 🎓 **Course Recommendations**: Personalized learning paths for skill development
- 📈 **Analytics Dashboard**: Comprehensive insights and trends
- 🔄 **Real-time Processing**: Instant analysis and recommendations

---

## 🌟 Core Features

### 📋 Resume Analysis
- 🔍 **Intelligent Parsing**: Extract skills, experience, education, and certifications
- 📊 **ATS Scoring**: Evaluate resume compatibility with Applicant Tracking Systems
- 🏷️ **Skill Detection**: Identify and categorize 12+ skill categories
- 📈 **Analytics**: Visualize resume strengths and improvement areas
- 📁 **Multi-Format Support**: PDF, DOCX, and TXT file support

### 💼 Job Matching Engine
- 🎯 **Smart Matching**: Match resumes with relevant job opportunities
- 🔗 **Real-time Data**: Integration with Naukri Jobs Dataset (Hugging Face)
- 📊 **Compatibility Scoring**: Calculate match percentage between candidates and positions
- 🔍 **Gap Analysis**: Identify skill gaps and improvement areas
- 💡 **Relevance Ranking**: ML-enhanced job ranking system

### 🎓 Course Recommendations
- 📚 **Personalized Learning**: AI-driven course suggestions based on skill gaps
- 🎯 **Target-Based**: Recommendations aligned with career goals
- 🏆 **Multi-Platform**: Courses from Coursera, Udemy, edX, and Pluralsight
- ⭐ **Quality Metrics**: Rating, duration, and enrollment information
- 🔄 **Continuous Learning**: Track progress and suggest next steps

### 🤖 ML-Enhanced Features
- 🧠 **Advanced Analytics**: Machine learning models for deeper insights
- 📊 **Trend Analysis**: Industry skill trends and market insights
- 🎲 **Predictive Scoring**: Predict job match success rates
- 📈 **Skill Trends**: Monitor emerging and in-demand skills

---

## 🛠️ Tech Stack

### Backend
| Technology | Version | Purpose |
|-----------|---------|----------|
| **FastAPI** | 0.100+ | REST API Framework |
| **Uvicorn** | Latest | ASGI Server |
| **Python** | 3.8+ | Backend Language |
| **scikit-learn** | 1.0+ | ML Models |
| **NLTK** | 3.8+ | NLP Processing |
| **Gensim** | 4.3+ | Text Analysis |
| **PyPDF2** | 3.0+ | PDF Processing |
| **python-docx** | 1.1+ | DOCX Processing |
| **NumPy** | 1.21+ | Numerical Computing |
| **Requests** | 2.31+ | HTTP Client |
| **BeautifulSoup4** | 4.12+ | Web Scraping |

### Frontend
| Technology | Version | Purpose |
|-----------|---------|----------|
| **React** | 18.3+ | UI Framework |
| **TypeScript** | 5.8+ | Type Safety |
| **Vite** | 7.1+ | Build Tool |
| **TailwindCSS** | 3.4+ | Styling |
| **shadcn/ui** | Latest | UI Components |
| **Radix UI** | Latest | Accessible Components |
| **React Router** | 6.30+ | Routing |
| **React Query** | 5.83+ | State Management |
| **React Hook Form** | 7.61+ | Form Management |
| **Zod** | 3.25+ | Schema Validation |
| **Recharts** | 2.15+ | Data Visualization |
| **Lucide React** | 0.462+ | Icons |
| **Sonner** | 1.7+ | Toast Notifications |

### DevOps & Infrastructure
- 🐳 **Docker**: Containerization
- 🐳 **Docker Compose**: Multi-container orchestration
- 📦 **Git**: Version control

---

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 16+ (for frontend)
- **npm/bun**: Package manager
- **Docker** (optional): For containerized deployment
- **Git**: Version control

---

## 🚀 Quick Start

### Option 1: Local Development Setup

#### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload
```

✅ Backend API available at: `http://localhost:8000`

#### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
# or
bun install

# Start development server
npm run dev
# or
bun run dev
```

✅ Frontend available at: `http://localhost:5173`

### Option 2: Docker Deployment

```bash
# From the backend directory
cd backend

# Build and run containers
docker-compose up --build

# Or use the provided script
./run-docker.sh
```

✅ Services will be available at their respective ports

### Stop Docker Containers

```bash
./stop-docker.sh
```

---

## 📚 API Documentation

Once the backend is running, access interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Detailed Docs**: [COMPREHENSIVE_API_DOCUMENTATION.md](backend/COMPREHENSIVE_API_DOCUMENTATION.md)

### Main API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload-resume` | Upload and analyze resume |
| POST | `/score` | Score resume against job description |
| GET | `/jobs` | Get job recommendations |
| GET | `/skill-gap` | Analyze skill gaps |
| GET | `/courses` | Get course recommendations |
| GET | `/analytics` | Get comprehensive analytics |
| POST | `/detailed-analysis` | Get detailed analysis |

---

## 💻 Usage Examples

### Upload and Analyze Resume

```bash
curl -X POST "http://localhost:8000/upload-resume" \
  -F "file=@resume.pdf" \
  -F "job_description=Senior Software Engineer role"
```

### Score Resume Against Job Description

```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Your resume text here...",
    "job_description": "Job description here..."
  }'
```

### Get Job Recommendations

```bash
curl -X GET "http://localhost:8000/jobs?skills=python,react,aws"
```

### Get Skill Gap Analysis

```bash
curl -X GET "http://localhost:8000/skill-gap?userSkills=python,javascript&roleRequiredSkills=python,javascript,react,node.js"
```

### Get Course Recommendations

```bash
curl -X GET "http://localhost:8000/courses?skills=machine%20learning,python"
```

---

## 📁 Project Structure

```
RESUME-INTELLIGENCE-SYSTEM/
│
├── 📂 backend/
│   ├── main.py                          # FastAPI application entry point
│   ├── resume_parser.py                 # Resume parsing & skill extraction
│   ├── ats_scorer.py                    # ATS scoring algorithm
│   ├── job_matcher.py                   # Job matching engine
│   ├── skill_analyzer.py                # Skill gap analysis
│   ├── course_recommender.py            # Course recommendation engine
│   ├── ml_enhanced_analyzer.py          # ML-based analysis
│   ├── train_models_colab.py            # Model training script
│   ├── ml_models/                       # Trained ML models
│   ├── requirements.txt                 # Python dependencies
│   ├── Dockerfile                       # Docker configuration
│   ├── docker-compose.yml               # Docker Compose setup
│   ├── run-docker.sh                    # Docker startup script
│   ├── stop-docker.sh                   # Docker shutdown script
│   ├── COMPREHENSIVE_API_DOCUMENTATION.md
│   ├── processed_resumes.json           # Processed resume cache
│   ├── jobs_cache.json                  # Jobs data cache
│   ├── resume_analytics.json            # Analytics data
│   └── skill_trends.json                # Skill trends data
│
├── 📂 frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Layout.tsx               # Main layout wrapper
│   │   │   ├── FileUpload.tsx           # Resume upload component
│   │   │   ├── ProgressCircle.tsx       # Progress visualization
│   │   │   ├── StatCard.tsx             # Statistics card
│   │   │   ├── SkillBadge.tsx           # Skill badge component
│   │   │   └── ui/                      # shadcn/ui components
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx            # Main dashboard
│   │   │   ├── Jobs.tsx                 # Job recommendations
│   │   │   ├── SkillGap.tsx             # Skill gap analysis
│   │   │   ├── Courses.tsx              # Course recommendations
│   │   │   ├── MLTools.tsx              # ML tools & features
│   │   │   ├── Analytics.tsx            # Analytics dashboard
│   │   │   └── NotFound.tsx             # 404 page
│   │   ├── hooks/
│   │   │   └── use-mobile.tsx           # Mobile detection hook
│   │   ├── lib/
│   │   │   └── api.ts                   # API service client
│   │   ├── App.tsx                      # Main App component
│   │   └── main.tsx                     # Entry point
│   ├── public/                          # Static assets
│   ├── package.json                     # Node dependencies
│   ├── vite.config.ts                   # Vite configuration
│   ├── tailwind.config.ts               # TailwindCSS config
│   ├── tsconfig.json                    # TypeScript config
│   └── README.md                        # Frontend README
│
└── README.md                            # This file
```

---

## 🔧 Core Modules Explained

### Resume Parser (`resume_parser.py`)
- Extracts structured information from resumes (PDF, DOCX, TXT)
- Identifies contact info, education, experience, and projects
- Categorizes skills into 12+ categories
- Provides analytics on resume completeness

### ATS Scorer (`ats_scorer.py`)
- Evaluates resume compatibility with Applicant Tracking Systems
- Analyzes keyword density and relevance
- Provides structure and format scoring
- Supports job description-based scoring
- ML-enhanced similarity matching

### Job Matcher (`job_matcher.py`)
- Matches resumes with job opportunities
- Uses Naukri Jobs Dataset from Hugging Face
- Implements intelligent skill-based matching
- Calculates relevance scores
- Caches job data for performance

### Skill Analyzer (`skill_analyzer.py`)
- Analyzes skill gaps between user and job requirements
- Prioritizes skills based on industry demand
- Provides learning path recommendations
- Tracks skill trends and market insights

### Course Recommender (`course_recommender.py`)
- Recommends courses based on skill gaps
- Scrapes multiple course platforms
- Provides course details (rating, duration, price)
- Supports skill-based filtering

### ML Enhanced Analyzer (`ml_enhanced_analyzer.py`)
- Advanced text similarity analysis
- TF-IDF based document matching
- Skill matching using ML models
- Predictive scoring capabilities

---

## 🎨 Frontend Pages

### Dashboard (`/`)
- Resume upload interface
- Real-time analysis results
- ATS score visualization
- Extracted skills display
- Contact information extraction

### Jobs (`/jobs`)
- Job recommendations based on skills
- Job details and requirements
- Skill matching indicators
- Apply links

### Skill Gap (`/skill-gap`)
- Identify missing skills
- Priority-based skill recommendations
- Learning resources
- Skill development roadmap

### Courses (`/courses`)
- Personalized course recommendations
- Course details and ratings
- Enrollment information
- Platform variety

### ML Tools (`/ml-tools`)
- Advanced ML-based analysis
- Predictive scoring
- Trend analysis
- Market insights

### Analytics (`/analytics`)
- Comprehensive statistics
- Trend visualization
- Resume processing analytics
- Market insights

---

## 🧪 Testing

### Frontend Linting

```bash
cd frontend
npm run lint
```

### Backend Testing

```bash
cd backend
# Run with test resume
python main.py
```

### Test Resume

A sample test resume is available at `backend/test_resume.txt`

---

## 🔐 Environment Variables

Create a `.env` file in the backend directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# CORS Configuration
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Job Search Configuration (if using external APIs)
JOB_API_KEY=your_api_key_here
JOB_API_SECRET=your_api_secret_here

# Course Platform APIs (optional)
COURSERA_API_KEY=your_key_here
UDEMY_API_KEY=your_key_here
```

---

## 📊 Data Files

The system generates and maintains several data files:

- **`processed_resumes.json`**: Cache of analyzed resumes
- **`jobs_cache.json`**: Cached job listings
- **`resume_analytics.json`**: Resume analysis statistics
- **`skill_trends.json`**: Industry skill trends

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Guidelines
- Follow PEP 8 for Python code
- Follow ESLint rules for TypeScript/React
- Add tests for new features
- Update documentation
- Keep commits atomic and descriptive

---

## 📝 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 🎯 Support & Issues

Have questions or found a bug?

- 📧 **Email**: ashutoshd072@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/resume-intelligence-system/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/resume-intelligence-system/discussions)

---

## 🎓 College Project

This is a comprehensive college project demonstrating:

- ✅ **Full-Stack Development**: React + FastAPI
- ✅ **Machine Learning Integration**: scikit-learn, NLTK, Gensim
- ✅ **API Design**: RESTful architecture with FastAPI
- ✅ **Modern Frontend**: React 18, TypeScript, TailwindCSS
- ✅ **DevOps**: Docker, Docker Compose
- ✅ **Data Analysis**: Analytics and visualization
- ✅ **Web Scraping**: Course and job data collection
- ✅ **NLP**: Resume parsing and skill extraction

---

## 🗺️ Roadmap

### Phase 1 (Current) ✅
- [x] Resume parsing and analysis
- [x] ATS scoring
- [x] Job matching
- [x] Skill gap analysis
- [x] Course recommendations
- [x] Basic analytics

### Phase 2 (Planned) 🚧
- [ ] 🔐 User authentication and profiles
- [ ] 💾 Resume storage and versioning
- [ ] 📧 Email notifications
- [ ] 🌍 Multi-language support
- [ ] 📱 Mobile app (React Native)
- [ ] 🔄 Real-time job updates

### Phase 3 (Future) 📋
- [ ] 🎤 Interview preparation module
- [ ] 📈 Career progression tracking
- [ ] 🤖 AI chatbot for career guidance
- [ ] 🏆 Gamification and achievements
- [ ] 🔗 LinkedIn integration
- [ ] 💼 Recruiter dashboard

---

## 📊 Performance Metrics

- **Resume Analysis**: < 2 seconds
- **Job Matching**: < 1 second
- **Course Recommendations**: < 1 second
- **API Response Time**: < 500ms average
- **Frontend Load Time**: < 3 seconds

---

## 🙏 Acknowledgments

- **Hugging Face**: For the Naukri Jobs Dataset
- **Coursera, Udemy, edX, Pluralsight**: Course data sources
- **Open Source Community**: For amazing libraries and tools
- **College**: For the opportunity to work on this project

---

## 📞 Contact

**Ashutosh Das**
- 📧 Email: ashutoshd072@gmail.com
- 🔗 LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/ashutosh-das)
- 🐙 GitHub: [Your GitHub](https://github.com/Ash-D3v)

---

**Made with ❤️ by Ashutosh Das**


