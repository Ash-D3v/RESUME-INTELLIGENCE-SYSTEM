# main.py - Complete Resume Analysis & Job Matching System
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import re
import io
from datetime import datetime
import PyPDF2
import docx
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules
from resume_parser import ResumeParser
from ats_scorer import ATSScorer
from job_matcher import JobMatcher
from skill_analyzer import SkillAnalyzer
from course_recommender import CourseRecommender
# from ml_enhanced_analyzer import MLEnhancedAnalyzer

app = FastAPI(title="Resume Analysis & Job Matching API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
resume_parser = ResumeParser()
ats_scorer = ATSScorer()
job_matcher = JobMatcher()
skill_analyzer = SkillAnalyzer()
course_recommender = CourseRecommender()
# ml_analyzer = MLEnhancedAnalyzer()

# Data storage files
PROCESSED_RESUMES_FILE = "processed_resumes.json"
JOBS_CACHE_FILE = "jobs_cache.json"

# Pydantic models
class ScoreRequest(BaseModel):
    resume_text: str
    job_description: str = ""

# Helper functions for file processing
def convert_courses_to_dicts(courses_data):
    """Convert Course objects to dictionaries for JSON serialization"""
    if isinstance(courses_data, dict):
        converted = {}
        for key, value in courses_data.items():
            if isinstance(value, list):
                converted[key] = [course.to_dict() if hasattr(course, 'to_dict') else course for course in value]
            else:
                converted[key] = value
        return converted
    elif isinstance(courses_data, list):
        return [course.to_dict() if hasattr(course, 'to_dict') else course for course in courses_data]
    return courses_data

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc_file = io.BytesIO(file_content)
        doc = docx.Document(doc_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error reading DOCX: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading DOCX: {str(e)}")

def save_resume_data(data: Dict[str, Any]) -> None:
    """Save processed resume data to JSON file"""
    try:
        if os.path.exists(PROCESSED_RESUMES_FILE):
            with open(PROCESSED_RESUMES_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = {"resumes": []}
        
        existing_data["resumes"].append(data)
        
        with open(PROCESSED_RESUMES_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving resume data: {str(e)}")

# API Endpoints

@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    job_description: str = Query("", description="Optional job description for targeted analysis")
):
    """
    Upload resume file and get comprehensive analysis including:
    - Resume parsing and skill extraction
    - ATS scoring
    - Job matching
    - Skill gap analysis
    - Course recommendations
    """
    # Validate file type
    allowed_types = [
        "application/pdf", 
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain"
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail="Only PDF, DOCX, and TXT files are supported"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Extract text based on file type
        if file.content_type == "application/pdf":
            extracted_text = extract_text_from_pdf(file_content)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = extract_text_from_docx(file_content)
        else:  # text/plain
            extracted_text = file_content.decode('utf-8')
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="No text could be extracted from the file"
            )
        
        # 1. Parse resume and extract skills
        parsed_resume = resume_parser.parse_resume(extracted_text)
        extracted_skills = resume_parser.extract_skills(extracted_text)
        
        # Get all user skills as a flat list
        all_user_skills = []
        for skill_category, skills in extracted_skills.items():
            all_user_skills.extend([skill.lower() for skill in skills])
        
        # 2. Calculate ATS scores
        if job_description.strip():
            ats_scores = ats_scorer.score_resume_with_job_description(extracted_text, job_description)
            score_breakdown = ats_scorer.get_score_breakdown(extracted_text, job_description)
            
            # Extract required skills from job description for skill gap analysis
            job_skills = resume_parser.extract_skills(job_description)
            required_skills = []
            for skill_category, skills in job_skills.items():
                required_skills.extend([skill.lower() for skill in skills])
            
            # Skill gap analysis
            skill_gap_analysis = skill_analyzer.analyze_skill_gaps(all_user_skills, required_skills)
            
            # Get course recommendations for missing skills
            missing_skills = skill_gap_analysis.get("missing_skills", [])
            targeted_courses = course_recommender.get_course_recommendations(missing_skills[:5])
        else:
            ats_scores = ats_scorer.score_resume(extracted_text)
            score_breakdown = ats_scorer.get_score_breakdown(extracted_text, "")
            skill_gap_analysis = {"missing_skills": [], "matching_skills": all_user_skills}
            targeted_courses = []
        
        # 3. Get job recommendations
        job_matches = job_matcher.find_matching_jobs(all_user_skills[:10], limit=5)
        
        # 4. Get course recommendations for user's skills
        course_recommendations = course_recommender.get_course_recommendations(all_user_skills[:5])
        
        # 5. Get popular courses as additional recommendations
        popular_courses = course_recommender.get_popular_courses()
        
        # 6. Get skill trends
        skill_trends = skill_analyzer.get_skill_trends()
        
        # Prepare comprehensive response
        comprehensive_response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "file_size": len(file_content),
            
            # Resume Analysis
            "resume_analysis": {
                "extracted_text": extracted_text,
                "parsed_resume": parsed_resume,
                "extracted_skills": extracted_skills,
                "total_skills_found": len(all_user_skills),
                "skills_by_category": {category: len(skills) for category, skills in extracted_skills.items()}
            },
            
            # ATS Scoring
            "ats_analysis": {
                "scores": ats_scores,
                "score_breakdown": score_breakdown,
                "job_description_provided": bool(job_description.strip())
            },
            
            # Skill Gap Analysis
            "skill_gap_analysis": skill_gap_analysis,
            
            # Job Recommendations
            "job_recommendations": {
                "matching_jobs": job_matches,
                "total_matches": len(job_matches),
                "search_based_on_skills": all_user_skills[:10]
            },
            
            # Course Recommendations
            "course_recommendations": {
                "targeted_courses": convert_courses_to_dicts(targeted_courses),  # Based on missing skills
                "general_courses": convert_courses_to_dicts(course_recommendations),  # Based on user skills
                "popular_courses": popular_courses,
                "total_courses": len(targeted_courses) + len(course_recommendations)
            },
            
            # Additional Insights
            "insights": {
                "skill_trends": skill_trends,
                "recommendations": [
                    "Focus on improving ATS score by adding more relevant keywords" if ats_scores.get('overallScore', 0) < 70 else "Great ATS score! Your resume is well-optimized",
                    f"Consider learning {', '.join(skill_gap_analysis.get('missing_skills', [])[:3])} to match job requirements" if skill_gap_analysis.get('missing_skills') else "You have strong skill alignment",
                    f"Found {len(job_matches)} relevant job opportunities based on your skills"
                ]
            }
        }
        
        # Save to file for future reference
        save_resume_data({
            "filename": file.filename,
            "timestamp": comprehensive_response["timestamp"],
            "skills_found": len(all_user_skills),
            "ats_score": ats_scores.get('overallScore', 0),
            "job_matches": len(job_matches)
        })
        
        return JSONResponse(status_code=200, content=comprehensive_response)
        
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@app.post("/score")
async def score_resume(request: ScoreRequest):
    """
    Score resume against job description
    Returns: { overallScore, structureScore, keywordScore, formatScore }
    """
    try:
        if not request.resume_text.strip():
            raise HTTPException(status_code=400, detail="Resume text cannot be empty")
        
        # Calculate ATS scores
        if request.job_description.strip():
            # Enhanced scoring with job description context
            scores = ats_scorer.score_resume_with_job_description(
                request.resume_text, 
                request.job_description
            )
        else:
            # Standard ATS scoring
            scores = ats_scorer.score_resume(request.resume_text)
        
        return JSONResponse(status_code=200, content=scores)
        
    except Exception as e:
        logger.error(f"Error scoring resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scoring resume: {str(e)}")

@app.get("/jobs")
async def get_suggested_jobs(skills: str = Query(..., description="Comma-separated list of skills")):
    """
    Get suggested jobs based on user skills
    Returns: [ { title, relevance, requiredSkills, location, company } ]
    """
    try:
        skill_list = [skill.strip().lower() for skill in skills.split(",") if skill.strip()]
        
        if not skill_list:
            raise HTTPException(status_code=400, detail="At least one skill must be provided")
        
        # Get job matches
        suggested_jobs = job_matcher.find_matching_jobs(skill_list)
        
        return JSONResponse(status_code=200, content=suggested_jobs)
        
    except Exception as e:
        logger.error(f"Error finding jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error finding jobs: {str(e)}")

@app.get("/skill-gap")
async def analyze_skill_gap(
    userSkills: str = Query(..., description="Comma-separated user skills"),
    roleRequiredSkills: str = Query(..., description="Comma-separated required skills for role")
):
    """
    Analyze skill gap between user skills and role requirements
    Returns: list of missing skills with priority levels
    """
    try:
        user_skills_list = [skill.strip().lower() for skill in userSkills.split(",") if skill.strip()]
        required_skills_list = [skill.strip().lower() for skill in roleRequiredSkills.split(",") if skill.strip()]
        
        if not user_skills_list or not required_skills_list:
            raise HTTPException(
                status_code=400, 
                detail="Both user skills and required skills must be provided"
            )
        
        # Analyze skill gap
        skill_gap_analysis = skill_analyzer.analyze_skill_gaps(user_skills_list, required_skills_list)
        
        return JSONResponse(status_code=200, content=skill_gap_analysis)
        
    except Exception as e:
        logger.error(f"Error analyzing skill gap: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing skill gap: {str(e)}")

@app.get("/courses")
async def get_course_recommendations(skills: str = Query(..., description="Comma-separated list of skills to learn")):
    """
    Get course recommendations for specified skills
    Returns: [ { title, provider, url, rating, duration } ]
    """
    try:
        skill_list = [skill.strip() for skill in skills.split(",") if skill.strip()]
        
        if not skill_list:
            raise HTTPException(status_code=400, detail="At least one skill must be provided")
        
        # Get course recommendations
        course_recommendations = course_recommender.get_course_recommendations(skill_list)
        
        # Convert Course objects to dictionaries
        converted_recommendations = convert_courses_to_dicts(course_recommendations)
        
        return JSONResponse(status_code=200, content=converted_recommendations)
        
    except Exception as e:
        logger.error(f"Error getting course recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting course recommendations: {str(e)}")

@app.get("/analytics")
async def get_analytics():
    """Get comprehensive analytics from all processed data"""
    try:
        analytics_data = {
            "resume_analytics": resume_parser.get_analytics(),
            "job_market_analytics": job_matcher.get_market_analytics(),
            "skill_trends": skill_analyzer.get_skill_trends(),
            "course_popularity": course_recommender.get_popular_courses()
        }
        
        return JSONResponse(status_code=200, content=analytics_data)
        
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")

@app.post("/detailed-analysis")
async def get_detailed_analysis(request: ScoreRequest):
    """
    Get comprehensive analysis including scores, job matches, skill gaps, and course recommendations
    """
    try:
        if not request.resume_text.strip():
            raise HTTPException(status_code=400, detail="Resume text cannot be empty")
        
        # Parse resume and extract skills
        parsed_resume = resume_parser.parse_resume(request.resume_text)
        extracted_skills = resume_parser.extract_skills(request.resume_text)
        
        # Get all user skills as a flat list
        all_user_skills = []
        for skill_category, skills in extracted_skills.items():
            all_user_skills.extend([skill.lower() for skill in skills])
        
        # Calculate scores
        if request.job_description.strip():
            scores = ats_scorer.score_resume_with_job_description(
                request.resume_text, 
                request.job_description
            )
            # Extract required skills from job description
            job_skills = resume_parser.extract_skills(request.job_description)
            required_skills = []
            for skill_category, skills in job_skills.items():
                required_skills.extend([skill.lower() for skill in skills])
            
            # Analyze skill gap
            skill_gap = skill_analyzer.analyze_skill_gaps(all_user_skills, required_skills)
        else:
            scores = ats_scorer.score_resume(request.resume_text)
            skill_gap = {"missing_skills": [], "matching_skills": all_user_skills}
        
        # Get job recommendations
        job_matches = job_matcher.find_matching_jobs(all_user_skills[:10])  # Limit for performance
        
        # Get course recommendations for missing skills
        missing_skills = skill_gap.get("missing_skills", [])
        course_recommendations = []
        if missing_skills:
            course_recommendations = course_recommender.get_course_recommendations(missing_skills[:5])
        
        comprehensive_analysis = {
            "resume_analysis": {
                "parsed_resume": parsed_resume,
                "extracted_skills": extracted_skills,
                "total_skills": len(all_user_skills)
            },
            "ats_scores": scores,
            "skill_gap_analysis": skill_gap,
            "job_recommendations": job_matches,
            "course_recommendations": convert_courses_to_dicts(course_recommendations),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(status_code=200, content=comprehensive_analysis)
        
    except Exception as e:
        logger.error(f"Error in detailed analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in detailed analysis: {str(e)}")

@app.delete("/clear-data")
async def clear_all_data():
    """Clear all stored data"""
    try:
        files_to_clear = [PROCESSED_RESUMES_FILE, JOBS_CACHE_FILE]
        cleared_files = []
        
        for file_path in files_to_clear:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleared_files.append(file_path)
        
        return JSONResponse(content={
            "message": "Data cleared successfully",
            "cleared_files": cleared_files
        })
        
    except Exception as e:
        logger.error(f"Error clearing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")

# ML-Enhanced Endpoints

@app.post("/ml-enhanced-matching")
async def ml_enhanced_matching(
    user_skills: str = Query(..., description="Comma-separated user skills"),
    job_skills: str = Query(..., description="Comma-separated job skills")
):
    """Enhanced skill matching using ML algorithms (TF-IDF, Word2Vec, Classification)"""
    try:
        user_skills_list = [skill.strip() for skill in user_skills.split(",") if skill.strip()]
        job_skills_list = [skill.strip() for skill in job_skills.split(",") if skill.strip()]
        
        if not user_skills_list or not job_skills_list:
            raise HTTPException(status_code=400, detail="Both user_skills and job_skills are required")
        
        # Use ML-enhanced matching
        results = ml_analyzer.enhanced_skill_matching(user_skills_list, job_skills_list)
        
        return JSONResponse(content={
            "status": "success",
            "ml_enhanced_results": results,
            "user_skills": user_skills_list,
            "job_skills": job_skills_list,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in ML-enhanced matching: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML matching error: {str(e)}")

@app.post("/predict-job-fit")
async def predict_job_fit(
    user_skills: str = Query(..., description="Comma-separated user skills"),
    job_skills: str = Query(..., description="Comma-separated job skills")
):
    """Predict job fit probability using trained ML classifier"""
    try:
        user_skills_list = [skill.strip() for skill in user_skills.split(",") if skill.strip()]
        job_skills_list = [skill.strip() for skill in job_skills.split(",") if skill.strip()]
        
        if not user_skills_list or not job_skills_list:
            raise HTTPException(status_code=400, detail="Both user_skills and job_skills are required")
        
        # Predict job fit probability
        fit_probability = ml_analyzer.predict_job_fit_probability(user_skills_list, job_skills_list)
        
        # Also get traditional scoring for comparison
        traditional_score = job_matcher.calculate_relevance_score(user_skills_list, job_skills_list)
        
        return JSONResponse(content={
            "status": "success",
            "ml_job_fit_probability": round(fit_probability, 3),
            "traditional_relevance_score": round(traditional_score, 2),
            "recommendation": "High fit" if fit_probability > 0.7 else "Medium fit" if fit_probability > 0.4 else "Low fit",
            "user_skills": user_skills_list,
            "job_skills": job_skills_list,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in job fit prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Job fit prediction error: {str(e)}")

@app.get("/ml-job-recommendations")
async def ml_job_recommendations(
    user_skills: str = Query(..., description="Comma-separated user skills"),
    interaction_history: str = Query("", description="JSON string of user job interactions")
):
    """Get job recommendations using collaborative filtering"""
    try:
        user_skills_list = [skill.strip() for skill in user_skills.split(",") if skill.strip()]
        
        if not user_skills_list:
            raise HTTPException(status_code=400, detail="user_skills parameter is required")
        
        # Parse interaction history
        user_interactions = []
        if interaction_history:
            try:
                user_interactions = json.loads(interaction_history)
            except:
                logger.warning("Invalid interaction_history JSON, using empty list")
        
        # Get all jobs from job matcher
        all_jobs = job_matcher.jobs_data
        
        if not all_jobs:
            # Try to fetch jobs if not loaded
            job_matcher._fetch_jobs_from_dataset()
            all_jobs = job_matcher.jobs_data
        
        # Get ML-powered recommendations
        recommendations = ml_analyzer.collaborative_filtering_recommendations(
            user_skills_list, user_interactions, all_jobs, n_recommendations=10
        )
        
        return JSONResponse(content={
            "status": "success",
            "ml_recommendations": recommendations,
            "total_jobs_analyzed": len(all_jobs),
            "user_skills": user_skills_list,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in ML job recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML recommendations error: {str(e)}")

@app.post("/train-ml-models")
async def train_ml_models():
    """Train ML models with sample data (for demonstration)"""
    try:
        from ml_enhanced_analyzer import generate_sample_training_data
        
        # Generate training data
        logger.info("Generating sample training data...")
        training_data = generate_sample_training_data(200)
        
        # Sample texts for Word2Vec
        sample_texts = [
            "python machine learning data science pandas numpy scikit-learn",
            "javascript react node.js web development frontend backend",
            "java spring boot backend development database sql",
            "aws cloud computing devops docker kubernetes terraform",
            "sql database postgresql mysql data analysis statistics",
            "html css javascript frontend web development responsive",
            "git version control ci cd jenkins devops automation",
            "linux bash shell scripting system administration"
        ] * 25  # Repeat to have enough data
        
        # Train Word2Vec model
        logger.info("Training Word2Vec model...")
        ml_analyzer.train_word2vec_model(sample_texts)
        
        # Train job fit classifier
        logger.info("Training job fit classifier...")
        ml_analyzer.train_job_fit_classifier(training_data)
        
        # Train clustering model
        sample_jobs = []
        if job_matcher.jobs_data:
            sample_jobs = job_matcher.jobs_data[:50]  # Use first 50 jobs
        else:
            # Create sample jobs if no real data
            sample_jobs = [
                {'job_id': 1, 'title': 'Data Scientist', 'skills': ['python', 'machine learning', 'pandas', 'sql']},
                {'job_id': 2, 'title': 'Web Developer', 'skills': ['javascript', 'react', 'html', 'css', 'node.js']},
                {'job_id': 3, 'title': 'DevOps Engineer', 'skills': ['aws', 'docker', 'kubernetes', 'linux', 'jenkins']},
                {'job_id': 4, 'title': 'Backend Developer', 'skills': ['java', 'spring', 'sql', 'api', 'microservices']},
                {'job_id': 5, 'title': 'ML Engineer', 'skills': ['python', 'tensorflow', 'aws', 'docker', 'mlops']}
            ]
        
        cluster_results = ml_analyzer.cluster_job_profiles(sample_jobs, n_clusters=min(3, len(sample_jobs)))
        
        # Save all models
        ml_analyzer.save_models()
        
        return JSONResponse(content={
            "status": "success",
            "message": "ML models trained successfully",
            "training_data_size": len(training_data),
            "word2vec_vocabulary_size": len(ml_analyzer.word2vec_model.wv) if ml_analyzer.word2vec_model else 0,
            "jobs_clustered": len(sample_jobs),
            "cluster_analysis": cluster_results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error training ML models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML training error: {str(e)}")

@app.get("/ml-analytics")
async def ml_analytics():
    """Get ML model analytics and performance metrics"""
    try:
        analytics = {
            "ml_models_status": {
                "word2vec_model": ml_analyzer.word2vec_model is not None,
                "job_fit_classifier": ml_analyzer.job_fit_classifier is not None,
                "job_clusters": ml_analyzer.job_clusters is not None
            },
            "model_details": {}
        }
        
        # Word2Vec details
        if ml_analyzer.word2vec_model:
            analytics["model_details"]["word2vec"] = {
                "vocabulary_size": len(ml_analyzer.word2vec_model.wv),
                "vector_size": ml_analyzer.word2vec_model.wv.vector_size,
                "sample_similarities": {}
            }
            
            # Sample similarity calculations
            try:
                sample_skills = ["python", "javascript", "machine learning", "web development"]
                for skill in sample_skills:
                    if skill.replace(" ", "_") in ml_analyzer.word2vec_model.wv:
                        similar = ml_analyzer.word2vec_model.wv.most_similar(skill.replace(" ", "_"), topn=3)
                        analytics["model_details"]["word2vec"]["sample_similarities"][skill] = similar
            except:
                pass
        
        # Classifier details
        if ml_analyzer.job_fit_classifier:
            analytics["model_details"]["classifier"] = {
                "model_type": "RandomForestClassifier",
                "n_estimators": ml_analyzer.job_fit_classifier.n_estimators,
                "feature_importance_available": hasattr(ml_analyzer.job_fit_classifier, 'feature_importances_')
            }
        
        # Clustering details
        if ml_analyzer.job_clusters:
            analytics["model_details"]["clustering"] = {
                "n_clusters": ml_analyzer.job_clusters.n_clusters,
                "algorithm": "KMeans"
            }
        
        return JSONResponse(content={
            "status": "success",
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting ML analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML analytics error: {str(e)}")

@app.get("/")
async def root():
    """API root endpoint with documentation"""
    return {
        "message": "Resume Analysis & Job Matching API",
        "version": "1.0.0",
        "status": "ready",
        "endpoints": {
            "POST /upload-resume": "Upload resume file for parsing and skill extraction",
            "POST /score": "Score resume against job description",
            "GET /jobs?skills=": "Get job suggestions based on skills",
            "GET /skill-gap?userSkills=&roleRequiredSkills=": "Analyze skill gaps",
            "GET /courses?skills=": "Get course recommendations",
            "POST /detailed-analysis": "Get comprehensive analysis",
            "GET /analytics": "Get system analytics",
            "DELETE /clear-data": "Clear all stored data",
            "POST /ml-enhanced-matching": "ML-enhanced skill matching (TF-IDF, Word2Vec)",
            "POST /predict-job-fit": "Predict job fit probability using ML classifier",
            "GET /ml-job-recommendations": "Get ML-powered job recommendations",
            "POST /train-ml-models": "Train ML models with sample data",
            "GET /ml-analytics": "Get ML model analytics and performance"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "resume_parser": resume_parser is not None,
            "ats_scorer": ats_scorer is not None,
            "job_matcher": job_matcher is not None,
            "skill_analyzer": skill_analyzer is not None,
            "course_recommender": course_recommender is not None,
            "ml_analyzer": False  # ml_analyzer is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Resume Analysis & Job Matching API...")
    print("Access the API at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)