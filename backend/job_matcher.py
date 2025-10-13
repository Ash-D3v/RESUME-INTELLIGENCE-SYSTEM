# job_matcher.py - Job Matcher using Naukri Jobs Dataset
import requests
import json
import re
import os
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class JobMatcher:
    """
    Job Matcher that uses Naukri Jobs Dataset from Hugging Face
    to find relevant job matches based on user skills
    """
    
    def __init__(self):
        self.jobs_cache_file = "jobs_cache.json"
        self.cache_duration_hours = 24  # Cache data for 24 hours
        self.dataset_url = "https://datasets-server.huggingface.co/rows?dataset=muhammetakkurt%2Fnaukri-jobs-dataset&config=default&split=train"
        self.jobs_data = []
        self.last_fetch_time = None
        
        # Load cached data if available
        self._load_cached_jobs()
    
    def _load_cached_jobs(self):
        """Load jobs from cache if available and not expired"""
        try:
            if os.path.exists(self.jobs_cache_file):
                with open(self.jobs_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Check if cache is still valid
                cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
                if datetime.now() - cache_time < timedelta(hours=self.cache_duration_hours):
                    self.jobs_data = cache_data.get('jobs', [])
                    self.last_fetch_time = cache_time
                    logger.info(f"Loaded {len(self.jobs_data)} jobs from cache")
                    return
        except Exception as e:
            logger.error(f"Error loading cached jobs: {e}")
        
        # If no valid cache, fetch fresh data
        self._fetch_jobs_from_dataset()
    
    def _fetch_jobs_from_dataset(self):
        """Fetch jobs data from Hugging Face dataset"""
        try:
            logger.info("Fetching jobs from Naukri dataset...")
            
            # Fetch data from Hugging Face dataset API
            response = requests.get(self.dataset_url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            rows = data.get('rows', [])
            
            self.jobs_data = []
            for row in rows:
                row_data = row.get('row', {})
                
                # Extract and clean job data
                job = {
                    'title': self._clean_text(row_data.get('Job Title', '')),
                    'company': self._clean_text(row_data.get('Company', '')),
                    'location': self._clean_text(row_data.get('Location', '')),
                    'experience': self._clean_text(row_data.get('Experience', '')),
                    'salary': self._clean_text(row_data.get('Salary', '')),
                    'skills': self._extract_skills_from_text(row_data.get('Skills', '')),
                    'description': self._clean_text(row_data.get('Job Description', '')),
                    'qualification': self._clean_text(row_data.get('Qualification', '')),
                    'job_id': row_data.get('Unnamed: 0', len(self.jobs_data))  # Use index as ID
                }
                
                # Only add jobs with meaningful data
                if job['title'] and job['company']:
                    self.jobs_data.append(job)
            
            # Cache the data
            self._cache_jobs_data()
            self.last_fetch_time = datetime.now()
            
            logger.info(f"Successfully fetched and cached {len(self.jobs_data)} jobs")
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching jobs: {e}")
            # Set empty data if network fails
            self.jobs_data = []
        except Exception as e:
            logger.error(f"Error fetching jobs: {e}")
            # Set empty data if other errors occur
            self.jobs_data = []
    
    def _cache_jobs_data(self):
        """Cache jobs data to file"""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'jobs': self.jobs_data
            }
            
            with open(self.jobs_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error caching jobs data: {e}")
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        if not text or text == 'nan':
            return ""
        return str(text).strip()
    
    def _extract_skills_from_text(self, skills_text):
        """Extract skills from skills text"""
        if not skills_text or skills_text == 'nan':
            return []
        
        # Clean and split skills
        skills_text = str(skills_text).lower()
        
        # Common separators for skills
        separators = [',', '|', ';', '/', '&', 'and', '\n', '\t']
        
        skills = [skills_text]
        for sep in separators:
            new_skills = []
            for skill in skills:
                new_skills.extend([s.strip() for s in skill.split(sep) if s.strip()])
            skills = new_skills
        
        # Filter out very short or common words
        filtered_skills = []
        stop_words = {'and', 'or', 'the', 'in', 'at', 'to', 'for', 'of', 'with', 'on', 'by'}
        
        for skill in skills:
            skill = skill.strip()
            if len(skill) >= 2 and skill not in stop_words:
                filtered_skills.append(skill)
        
        return list(set(filtered_skills))[:20]  # Limit to 20 skills per job
    

    
    def calculate_relevance_score(self, user_skills: List[str], job_skills: List[str], use_ml: bool = False) -> float:
        """Calculate relevance score between user skills and job requirements"""
        if not job_skills or not user_skills:
            return 0.0
        
        # Use ML-enhanced matching if available
        if use_ml:
            try:
                from ml_enhanced_analyzer import MLEnhancedAnalyzer
                ml_analyzer = MLEnhancedAnalyzer()
                results = ml_analyzer.enhanced_skill_matching(user_skills, job_skills)
                # Use overall_match_score if available, otherwise fall back to tfidf_similarity
                overall_score = results.get('overall_match_score', 0.0)
                if overall_score > 0:
                    return overall_score
                return results.get('tfidf_similarity', 0.0) * 100
            except Exception as e:
                logger.warning(f"ML matching failed, falling back to rule-based: {e}")
                # Continue to fallback
        
        # Fallback to original rule-based approach
        user_skills_lower = [skill.lower().strip() for skill in user_skills]
        job_skills_lower = [skill.lower().strip() for skill in job_skills]
        
        # Direct matches
        direct_matches = len(set(user_skills_lower) & set(job_skills_lower))
        
        # Partial matches (substring matching)
        partial_matches = 0
        for user_skill in user_skills_lower:
            for job_skill in job_skills_lower:
                if user_skill in job_skill or job_skill in user_skill:
                    partial_matches += 0.5
        
        # Calculate score
        total_matches = direct_matches + partial_matches
        
        if len(job_skills_lower) == 0:
            return 0.0
        
        relevance_score = (total_matches / len(job_skills_lower)) * 100
        return min(relevance_score, 100.0)
    
    def find_matching_jobs(self, user_skills: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Find jobs matching user skills"""
        if not self.jobs_data:
            self._fetch_jobs_from_dataset()
        
        # If still no data after fetch, use fallback mock data
        if not self.jobs_data:
            logger.warning("No job data available, using fallback mock data")
            self.jobs_data = self._get_fallback_jobs()
        
        if not user_skills:
            return []
        
        matching_jobs = []
        
        for job in self.jobs_data:
            relevance_score = self.calculate_relevance_score(user_skills, job['skills'])
            
            if relevance_score > 5:  # Minimum relevance threshold (lowered for better matching)
                matching_job = {
                    'title': job['title'],
                    'company': job['company'],
                    'location': job['location'],
                    'experience': job['experience'],
                    'salary': job['salary'],
                    'relevance': round(relevance_score, 2),
                    'requiredSkills': job['skills'][:10],  # Limit to 10 skills for display
                    'description': job['description'][:200] + "..." if len(job['description']) > 200 else job['description'],
                    'qualification': job['qualification'],
                    'job_id': job['job_id'],
                    'matching_skills': list(set([skill.lower() for skill in user_skills]) & 
                                           set([skill.lower() for skill in job['skills']]))
                }
                matching_jobs.append(matching_job)
        
        # Sort by relevance score (descending)
        matching_jobs.sort(key=lambda x: x['relevance'], reverse=True)
        
        return matching_jobs[:limit]
    
    def _get_fallback_jobs(self) -> List[Dict[str, Any]]:
        """Provide fallback mock jobs when external dataset is unavailable"""
        return [
            {
                'title': 'Senior Python Developer',
                'company': 'Tech Innovations Inc',
                'location': 'Mumbai, Maharashtra',
                'experience': '3-5 years',
                'salary': '15-25 LPA',
                'skills': ['python', 'django', 'flask', 'postgresql', 'docker', 'aws', 'rest api'],
                'description': 'We are looking for an experienced Python developer to join our team. You will be responsible for building scalable backend services.',
                'qualification': 'BTech/BE in Computer Science or related field',
                'job_id': 1001
            },
            {
                'title': 'Full Stack JavaScript Developer',
                'company': 'Digital Solutions Ltd',
                'location': 'Bangalore, Karnataka',
                'experience': '2-4 years',
                'salary': '12-20 LPA',
                'skills': ['javascript', 'react', 'node.js', 'express', 'mongodb', 'typescript', 'git'],
                'description': 'Join our dynamic team to build modern web applications using the latest JavaScript technologies.',
                'qualification': 'BTech/MCA in relevant field',
                'job_id': 1002
            },
            {
                'title': 'Data Scientist',
                'company': 'Analytics Pro',
                'location': 'Pune, Maharashtra',
                'experience': '2-5 years',
                'salary': '18-30 LPA',
                'skills': ['python', 'machine learning', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'sql', 'data visualization'],
                'description': 'Work on cutting-edge ML projects and help drive data-driven decision making.',
                'qualification': 'MTech/MS in Data Science, Statistics, or related field',
                'job_id': 1003
            },
            {
                'title': 'DevOps Engineer',
                'company': 'Cloud Systems Corp',
                'location': 'Hyderabad, Telangana',
                'experience': '3-6 years',
                'salary': '16-28 LPA',
                'skills': ['aws', 'docker', 'kubernetes', 'jenkins', 'terraform', 'linux', 'python', 'ci/cd'],
                'description': 'Manage and optimize our cloud infrastructure and deployment pipelines.',
                'qualification': 'BTech in Computer Science or equivalent',
                'job_id': 1004
            },
            {
                'title': 'Frontend React Developer',
                'company': 'UI/UX Innovations',
                'location': 'Chennai, Tamil Nadu',
                'experience': '1-3 years',
                'salary': '8-15 LPA',
                'skills': ['react', 'javascript', 'html', 'css', 'typescript', 'redux', 'tailwindcss'],
                'description': 'Create beautiful and responsive user interfaces for our web applications.',
                'qualification': 'BTech/BCA in relevant field',
                'job_id': 1005
            },
            {
                'title': 'Backend Java Developer',
                'company': 'Enterprise Solutions',
                'location': 'Gurgaon, Haryana',
                'experience': '4-7 years',
                'salary': '20-35 LPA',
                'skills': ['java', 'spring boot', 'microservices', 'sql', 'kafka', 'redis', 'aws'],
                'description': 'Design and develop enterprise-grade backend systems using Java and Spring.',
                'qualification': 'BTech/MTech in Computer Science',
                'job_id': 1006
            },
            {
                'title': 'Machine Learning Engineer',
                'company': 'AI Research Labs',
                'location': 'Bangalore, Karnataka',
                'experience': '3-6 years',
                'salary': '25-40 LPA',
                'skills': ['python', 'tensorflow', 'pytorch', 'deep learning', 'nlp', 'computer vision', 'aws', 'mlops'],
                'description': 'Build and deploy state-of-the-art ML models for production systems.',
                'qualification': 'MTech/PhD in Machine Learning, AI, or related field',
                'job_id': 1007
            },
            {
                'title': 'Mobile App Developer (React Native)',
                'company': 'Mobile First Tech',
                'location': 'Mumbai, Maharashtra',
                'experience': '2-4 years',
                'salary': '12-22 LPA',
                'skills': ['react native', 'javascript', 'typescript', 'ios', 'android', 'redux', 'firebase'],
                'description': 'Develop cross-platform mobile applications for iOS and Android.',
                'qualification': 'BTech/BE in Computer Science',
                'job_id': 1008
            },
            {
                'title': 'Cloud Architect',
                'company': 'Cloud Native Solutions',
                'location': 'Pune, Maharashtra',
                'experience': '5-8 years',
                'salary': '30-50 LPA',
                'skills': ['aws', 'azure', 'gcp', 'kubernetes', 'terraform', 'microservices', 'system design'],
                'description': 'Design and implement scalable cloud architectures for enterprise clients.',
                'qualification': 'BTech/MTech with cloud certifications',
                'job_id': 1009
            },
            {
                'title': 'QA Automation Engineer',
                'company': 'Quality Assurance Inc',
                'location': 'Noida, Uttar Pradesh',
                'experience': '2-5 years',
                'salary': '10-18 LPA',
                'skills': ['selenium', 'python', 'java', 'javascript', 'cypress', 'api testing', 'ci/cd'],
                'description': 'Build and maintain automated testing frameworks for web and mobile applications.',
                'qualification': 'BTech/BE in Computer Science',
                'job_id': 1010
            }
        ]
    
    def get_jobs_by_location(self, location: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get jobs filtered by location"""
        if not self.jobs_data:
            self._fetch_jobs_from_dataset()
        
        location_lower = location.lower()
        matching_jobs = []
        
        for job in self.jobs_data:
            if location_lower in job['location'].lower():
                matching_jobs.append({
                    'title': job['title'],
                    'company': job['company'],
                    'location': job['location'],
                    'experience': job['experience'],
                    'salary': job['salary'],
                    'skills': job['skills'][:10],
                    'job_id': job['job_id']
                })
        
        return matching_jobs[:limit]
    
    def get_jobs_by_experience(self, min_experience: int, max_experience: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Get jobs filtered by experience range"""
        if not self.jobs_data:
            self._fetch_jobs_from_dataset()
        
        matching_jobs = []
        
        for job in self.jobs_data:
            # Extract experience numbers from experience text
            experience_text = job['experience'].lower()
            experience_numbers = re.findall(r'\d+', experience_text)
            
            if experience_numbers:
                job_min_exp = int(experience_numbers[0])
                job_max_exp = int(experience_numbers[-1]) if len(experience_numbers) > 1 else job_min_exp
                
                # Check if there's overlap between user experience and job requirements
                if not (max_experience < job_min_exp or min_experience > job_max_exp):
                    matching_jobs.append({
                        'title': job['title'],
                        'company': job['company'],
                        'location': job['location'],
                        'experience': job['experience'],
                        'salary': job['salary'],
                        'skills': job['skills'][:10],
                        'job_id': job['job_id']
                    })
        
        return matching_jobs[:limit]
    
    def get_trending_skills(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most in-demand skills from job postings"""
        if not self.jobs_data:
            self._fetch_jobs_from_dataset()
        
        all_skills = []
        for job in self.jobs_data:
            all_skills.extend(job['skills'])
        
        skill_counts = Counter(all_skills)
        trending_skills = []
        
        for skill, count in skill_counts.most_common(limit):
            trending_skills.append({
                'skill': skill,
                'demand_count': count,
                'percentage': round((count / len(self.jobs_data)) * 100, 2)
            })
        
        return trending_skills
    
    def get_salary_insights(self, skill: str) -> Dict[str, Any]:
        """Get salary insights for a specific skill"""
        if not self.jobs_data:
            self._fetch_jobs_from_dataset()
        
        skill_lower = skill.lower()
        relevant_jobs = []
        
        for job in self.jobs_data:
            if skill_lower in [s.lower() for s in job['skills']]:
                salary_text = job['salary']
                # Extract salary numbers (simplified)
                salary_numbers = re.findall(r'\d+', salary_text)
                if salary_numbers:
                    # Take average if range is provided
                    avg_salary = sum(int(num) for num in salary_numbers[:2]) / len(salary_numbers[:2])
                    relevant_jobs.append({
                        'title': job['title'],
                        'company': job['company'],
                        'location': job['location'],
                        'salary_numeric': avg_salary,
                        'salary_text': salary_text
                    })
        
        if not relevant_jobs:
            return {'message': f'No salary data available for {skill}'}
        
        salaries = [job['salary_numeric'] for job in relevant_jobs]
        
        return {
            'skill': skill,
            'job_count': len(relevant_jobs),
            'average_salary': round(sum(salaries) / len(salaries), 2),
            'min_salary': min(salaries),
            'max_salary': max(salaries),
            'sample_jobs': relevant_jobs[:5]
        }
    
    def get_market_analytics(self) -> Dict[str, Any]:
        """Get overall job market analytics"""
        if not self.jobs_data:
            self._fetch_jobs_from_dataset()
        
        if not self.jobs_data:
            return {'message': 'No job data available'}
        
        # Location distribution
        locations = [job['location'] for job in self.jobs_data if job['location']]
        location_counts = Counter(locations)
        
        # Company distribution
        companies = [job['company'] for job in self.jobs_data if job['company']]
        company_counts = Counter(companies)
        
        # Experience level distribution
        experience_levels = []
        for job in self.jobs_data:
            exp_text = job['experience'].lower()
            if 'fresher' in exp_text or '0' in exp_text:
                experience_levels.append('Fresher')
            elif any(num in exp_text for num in ['1', '2']):
                experience_levels.append('Junior (1-2 years)')
            elif any(num in exp_text for num in ['3', '4', '5']):
                experience_levels.append('Mid-level (3-5 years)')
            else:
                experience_levels.append('Senior (5+ years)')
        
        exp_level_counts = Counter(experience_levels)
        
        return {
            'total_jobs': len(self.jobs_data),
            'last_updated': self.last_fetch_time.isoformat() if self.last_fetch_time else None,
            'top_locations': dict(location_counts.most_common(10)),
            'top_companies': dict(company_counts.most_common(10)),
            'experience_distribution': dict(exp_level_counts),
            'trending_skills': self.get_trending_skills(10),
            'data_freshness': 'Good' if self.last_fetch_time and 
                            datetime.now() - self.last_fetch_time < timedelta(hours=self.cache_duration_hours) 
                            else 'Stale'
        }
    
    def search_jobs(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search jobs by title, company, or description"""
        if not self.jobs_data:
            self._fetch_jobs_from_dataset()
        
        query_lower = query.lower()
        matching_jobs = []
        
        for job in self.jobs_data:
            # Search in title, company, description
            search_text = f"{job['title']} {job['company']} {job['description']}".lower()
            
            if query_lower in search_text:
                matching_jobs.append({
                    'title': job['title'],
                    'company': job['company'],
                    'location': job['location'],
                    'experience': job['experience'],
                    'salary': job['salary'],
                    'skills': job['skills'][:10],
                    'description': job['description'][:200] + "..." if len(job['description']) > 200 else job['description'],
                    'job_id': job['job_id']
                })
        
        return matching_jobs[:limit]
    
    def refresh_jobs_cache(self) -> Dict[str, str]:
        """Force refresh the jobs cache"""
        try:
            self._fetch_jobs_from_dataset()
            return {
                'status': 'success',
                'message': f'Successfully refreshed {len(self.jobs_data)} jobs',
                'last_updated': self.last_fetch_time.isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to refresh cache: {str(e)}'
            }
            