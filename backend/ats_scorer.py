# ats_scorer.py - Enhanced ATS Scorer with Job Description Matching
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Dict, List, Set
from collections import Counter
import json
from datetime import datetime

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
    except:
        # Fallback if NLTK download fails
        pass

class ATSScorer:
    """
    Enhanced ATS (Applicant Tracking System) Scorer
    Evaluates resume quality based on ATS-friendly criteria and job description matching
    """
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stop words if NLTK fails
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
        
        # Enhanced industry-standard keywords that ATS systems commonly look for
        self.high_value_keywords = {
            'technical_skills': [
                'python', 'java', 'javascript', 'react', 'node', 'sql', 'html', 'css',
                'aws', 'docker', 'kubernetes', 'git', 'agile', 'scrum', 'api', 'rest',
                'mongodb', 'postgresql', 'mysql', 'linux', 'windows', 'azure', 'gcp',
                'tensorflow', 'pytorch', 'machine learning', 'data science', 'analytics',
                'spring', 'django', 'flask', 'angular', 'vue', 'typescript', 'php', 'ruby',
                'go', 'rust', 'kotlin', 'swift', 'c++', 'c#', '.net', 'jenkins', 'ci/cd',
                'microservices', 'devops', 'cloud computing', 'big data', 'hadoop', 'spark'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
                'creative', 'innovative', 'collaborative', 'strategic', 'detail oriented',
                'organized', 'adaptable', 'critical thinking', 'time management',
                'project management', 'mentoring', 'training', 'presentation',
                'cross-functional', 'stakeholder management', 'client relations'
            ],
            'action_verbs': [
                'developed', 'implemented', 'designed', 'created', 'managed', 'led',
                'improved', 'optimized', 'achieved', 'delivered', 'collaborated',
                'analyzed', 'established', 'executed', 'coordinated', 'streamlined',
                'built', 'architected', 'deployed', 'maintained', 'automated',
                'integrated', 'enhanced', 'reduced', 'increased', 'accelerated',
                'spearheaded', 'initiated', 'transformed', 'modernized', 'scaled'
            ],
            'certifications': [
                'certified', 'certification', 'licensed', 'accredited', 'diploma',
                'degree', 'bachelor', 'master', 'phd', 'mba', 'pmp', 'cissp', 'cpa',
                'aws certified', 'azure certified', 'google cloud', 'scrum master',
                'oracle certified', 'microsoft certified', 'cisco certified',
                'kubernetes certified', 'docker certified', 'terraform certified'
            ]
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_keywords_from_job_description(self, job_description: str) -> Dict[str, List[str]]:
        """Extract important keywords from job description"""
        if not job_description.strip():
            return {}
        
        cleaned_text = self.clean_text(job_description)
        job_keywords = {
            'technical_skills': [],
            'soft_skills': [],
            'action_verbs': [],
            'certifications': [],
            'specific_requirements': []
        }
        
        # Extract keywords from job description using our predefined lists
        for category, keywords in self.high_value_keywords.items():
            found_keywords = []
            for keyword in keywords:
                if keyword in cleaned_text:
                    found_keywords.append(keyword)
            job_keywords[category] = found_keywords
        
        # Extract specific requirements (requirements, qualifications, etc.)
        requirement_patterns = [
            r'(?:require[ds]?|must have|need|essential|mandatory)[\s\w]*?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
            r'(?:experience with|knowledge of|proficient in|familiar with)[\s\w]*?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
            r'(?:bachelor|master|degree in)[\s\w]*?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
        ]
        
        specific_reqs = []
        for pattern in requirement_patterns:
            matches = re.findall(pattern, job_description, re.IGNORECASE)
            specific_reqs.extend([match.strip().lower() for match in matches if len(match.strip()) > 2])
        
        job_keywords['specific_requirements'] = list(set(specific_reqs))[:10]  # Limit to top 10
        
        return job_keywords
    
    def calculate_keyword_match_score(self, resume_text: str, job_keywords: Dict[str, List[str]], use_ml: bool = True) -> int:
        """Calculate how well resume keywords match job description keywords using ML-enhanced similarity"""
        if not job_keywords or not resume_text.strip():
            return 0
        
        # Use ML-enhanced text similarity if available
        if use_ml:
            try:
                from ml_enhanced_analyzer import MLEnhancedAnalyzer
                ml_analyzer = MLEnhancedAnalyzer()
                
                # Convert job_keywords dict to flat text for comparison
                job_text = " ".join([" ".join(keywords) for keywords in job_keywords.values()])
                
                # Use TF-IDF similarity for document matching
                similarity_score = ml_analyzer.calculate_tfidf_similarity(resume_text, job_text)
                
                # Convert similarity to ATS score (0-100)
                ats_score = int(similarity_score * 100)
                return min(ats_score, 100)
                
            except Exception as e:
                logger.warning(f"ML-enhanced ATS scoring failed, falling back to rule-based: {e}")
        
        # Fallback to original rule-based approach
        cleaned_resume = self.clean_text(resume_text)
        total_score = 0
        total_weight = 0
        
        # Weight different categories based on importance for job matching
        category_weights = {
            'technical_skills': 3.0,  # Most important for matching
            'specific_requirements': 2.5,
            'action_verbs': 2.0,
            'soft_skills': 1.5,
            'certifications': 2.0
        }
        
        for category, keywords in job_keywords.items():
            if not keywords:
                continue
                
            matches = 0
            for keyword in keywords:
                if keyword.lower() in cleaned_resume:
                    matches += 1
            
            weight = category_weights.get(category, 1.0)
            category_score = (matches / len(keywords)) * 100 if keywords else 0
            total_score += category_score * weight
            total_weight += weight
        
        if total_weight > 0:
            return min(int(total_score / total_weight), 100)
        return 0
    
    def calculate_keyword_score(self, resume_text: str) -> int:
        """
        Calculate ATS keyword density and relevance score (original method)
        Returns score from 0-100 based on presence of industry-relevant keywords
        """
        if not resume_text.strip():
            return 0
            
        cleaned_text = self.clean_text(resume_text)
        score = 0
        total_weight = 0
        
        # Check for high-value keywords across categories
        for category, keywords in self.high_value_keywords.items():
            category_matches = 0
            for keyword in keywords:
                if keyword in cleaned_text:
                    category_matches += 1
            
            # Weight different categories based on ATS importance
            if category == 'technical_skills':
                weight = 2.5  # Technical skills weighted highest
                score += category_matches * weight
                total_weight += len(keywords) * weight
            elif category == 'action_verbs':
                weight = 2.0  # Action verbs very important for ATS
                score += category_matches * weight
                total_weight += len(keywords) * weight
            elif category == 'soft_skills':
                weight = 1.5
                score += category_matches * weight
                total_weight += len(keywords) * weight
            else:  # certifications
                weight = 1.0
                score += category_matches * weight
                total_weight += len(keywords) * weight
        
        # Calculate percentage but ensure reasonable scaling
        if total_weight > 0:
            keyword_density = (score / total_weight) * 100
            return min(int(keyword_density), 100)
        return 0
    
    def calculate_structure_score(self, resume_text: str) -> int:
        """
        Evaluate ATS-friendly resume structure
        Returns score from 0-100 based on presence of essential resume sections
        """
        if not resume_text.strip():
            return 0
            
        score = 0
        text_lower = resume_text.lower()
        
        # Essential sections for ATS parsing with their importance weights
        essential_sections = {
            'contact': {
                'keywords': ['email', 'phone', 'address', 'linkedin', 'contact'],
                'weight': 25,  # Critical for ATS
                'required': True
            },
            'experience': {
                'keywords': ['experience', 'employment', 'work history', 'professional experience', 'career'],
                'weight': 25,  # Critical for ATS
                'required': True
            },
            'education': {
                'keywords': ['education', 'degree', 'university', 'college', 'school', 'bachelor', 'master'],
                'weight': 15,
                'required': False
            },
            'skills': {
                'keywords': ['skills', 'technical skills', 'competencies', 'proficiencies', 'technologies'],
                'weight': 20,
                'required': False
            },
            'summary': {
                'keywords': ['summary', 'profile', 'objective', 'about', 'professional summary'],
                'weight': 10,
                'required': False
            }
        }
        
        # Check for presence of each section
        for section_type, section_info in essential_sections.items():
            found = any(keyword in text_lower for keyword in section_info['keywords'])
            if found:
                score += section_info['weight']
            elif section_info['required']:
                score -= 10  # Penalty for missing critical sections
        
        # Validate contact information patterns
        contact_bonus = 0
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_patterns = [
            r'(\+?\d{1,3}[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}'
        ]
        
        if re.search(email_pattern, resume_text):
            contact_bonus += 5
        
        phone_found = any(re.search(pattern, resume_text) for pattern in phone_patterns)
        if phone_found:
            contact_bonus += 5
        
        score += contact_bonus
        
        # Check for employment dates (critical for ATS chronological parsing)
        date_patterns = [
            r'\b(19|20)\d{2}\b',  # 4-digit years
            r'\b\d{1,2}/\d{4}\b',  # MM/YYYY
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (19|20)\d{2}\b',  # Month Year
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December) (19|20)\d{2}\b'
        ]
        
        total_dates = 0
        for pattern in date_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            total_dates += len(matches)
        
        # Award points based on date presence (indicates employment history)
        if total_dates >= 4:  # Multiple employment periods with dates
            score += 15
        elif total_dates >= 2:  # At least some dates present
            score += 10
        elif total_dates >= 1:
            score += 5
        
        return max(min(score, 100), 0)
    
    def calculate_format_score(self, resume_text: str) -> int:
        """
        Evaluate ATS-friendly formatting
        Returns score from 0-100 based on formatting that ATS systems can parse easily
        """
        if not resume_text.strip():
            return 0
            
        score = 0
        
        # Optimal length for ATS parsing (not too short, not too long)
        word_count = len(resume_text.split())
        if 300 <= word_count <= 800:  # Sweet spot for ATS
            score += 25
        elif 200 <= word_count <= 1000:
            score += 20
        elif 150 <= word_count <= 1200:
            score += 15
        elif word_count >= 100:
            score += 10
        else:
            score += 5
        
        # Check for proper structure indicators
        lines = resume_text.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        # Sufficient content structure
        if len(non_empty_lines) >= 15:
            score += 15
        elif len(non_empty_lines) >= 10:
            score += 10
        elif len(non_empty_lines) >= 5:
            score += 5
        
        # Professional language patterns (action verbs usage)
        sentences = re.split(r'[.!?]+', resume_text)
        action_verb_usage = 0
        
        for sentence in sentences[:15]:  # Check first 15 sentences
            sentence_clean = self.clean_text(sentence)
            if any(verb in sentence_clean for verb in self.high_value_keywords['action_verbs']):
                action_verb_usage += 1
        
        if action_verb_usage >= 5:
            score += 20
        elif action_verb_usage >= 3:
            score += 15
        elif action_verb_usage >= 1:
            score += 10
        
        # Quantifiable achievements (numbers, percentages, metrics)
        quantification_patterns = [
            r'\b\d+%',  # Percentages
            r'\b\d+\s*(million|thousand|k\b|m\b)',  # Large numbers
            r'\$\d+',  # Currency
            r'\b\d+\s*(years?|months?|weeks?)',  # Time periods
            r'\b\d+\s*(projects?|clients?|users?|customers?)',  # Quantities
            r'\bincrease[d]?\s+.*\d+',  # Increases with numbers
            r'\breduced?\s+.*\d+',  # Reductions with numbers
            r'\bimproved?\s+.*\d+'  # Improvements with numbers
        ]
        
        total_quantifications = 0
        for pattern in quantification_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            total_quantifications += len(matches)
        
        if total_quantifications >= 8:
            score += 20
        elif total_quantifications >= 5:
            score += 15
        elif total_quantifications >= 3:
            score += 10
        elif total_quantifications >= 1:
            score += 5
        
        # Avoid formatting that confuses ATS
        problematic_elements = {
            'special_bullets': ['→', '•', '◆', '★', '▪', '◦'],
            'tables': ['|'],  # Table characters
            'excessive_formatting': ['***', '___', '###']
        }
        
        formatting_penalty = 0
        for category, chars in problematic_elements.items():
            if any(char in resume_text for char in chars):
                formatting_penalty += 3
        
        score -= min(formatting_penalty, 15)  # Cap penalty at 15 points
        
        # Reward consistent capitalization (proper nouns, sentence beginnings)
        proper_caps = len(re.findall(r'\b[A-Z][a-z]+\b', resume_text))
        if proper_caps >= 10:
            score += 10
        elif proper_caps >= 5:
            score += 5
        
        # Check for excessive repetition (indicates poor quality)
        words = self.clean_text(resume_text).split()
        if len(words) > 50:  # Only check if sufficient content
            word_counts = Counter(words)
            most_common = word_counts.most_common(1)[0] if word_counts else ('', 0)
            
            if most_common[1] > len(words) * 0.05:  # More than 5% repetition
                score -= 10
        
        return max(min(score, 100), 0)
    
    def calculate_overall_score(self, structure_score: int, keyword_score: int, format_score: int) -> int:
        """
        Calculate weighted overall ATS score
        Structure is most important (45%), then keywords (35%), then format (20%)
        """
        overall = int(
            structure_score * 0.45 + 
            keyword_score * 0.35 + 
            format_score * 0.20
        )
        return min(max(overall, 0), 100)
    
    def score_resume(self, resume_text: str) -> Dict[str, int]:
        """
        Main method to score a resume (original functionality)
        Returns dictionary with all scores
        """
        if not resume_text or not resume_text.strip():
            return {
                'overallScore': 0,
                'structureScore': 0,
                'keywordScore': 0,
                'formatScore': 0
            }
        
        keyword_score = self.calculate_keyword_score(resume_text)
        structure_score = self.calculate_structure_score(resume_text)
        format_score = self.calculate_format_score(resume_text)
        overall_score = self.calculate_overall_score(structure_score, keyword_score, format_score)
        
        return {
            'overallScore': overall_score,
            'structureScore': structure_score,
            'keywordScore': keyword_score,
            'formatScore': format_score
        }
    
    def score_resume_with_job_description(self, resume_text: str, job_description: str) -> Dict[str, int]:
        """
        Enhanced scoring method that considers job description matching
        Returns enhanced scores including job match score
        """
        if not resume_text or not resume_text.strip():
            return {
                'overallScore': 0,
                'structureScore': 0,
                'keywordScore': 0,
                'formatScore': 0,
                'jobMatchScore': 0
            }
        
        # Calculate standard scores
        structure_score = self.calculate_structure_score(resume_text)
        format_score = self.calculate_format_score(resume_text)
        
        # Calculate job-specific keyword matching
        if job_description.strip():
            job_keywords = self.extract_keywords_from_job_description(job_description)
            job_match_score = self.calculate_keyword_match_score(resume_text, job_keywords)
            
            # Enhanced keyword score considering both general keywords and job-specific ones
            general_keyword_score = self.calculate_keyword_score(resume_text)
            keyword_score = int((general_keyword_score * 0.4) + (job_match_score * 0.6))
        else:
            keyword_score = self.calculate_keyword_score(resume_text)
            job_match_score = 0
        
        # Enhanced overall score calculation
        if job_description.strip():
            # Give more weight to job matching when job description is provided
            overall_score = int(
                structure_score * 0.3 + 
                keyword_score * 0.25 + 
                format_score * 0.15 +
                job_match_score * 0.3
            )
        else:
            overall_score = self.calculate_overall_score(structure_score, keyword_score, format_score)
        
        return {
            'overallScore': min(max(overall_score, 0), 100),
            'structureScore': structure_score,
            'keywordScore': keyword_score,
            'formatScore': format_score,
            'jobMatchScore': job_match_score
        }
    
    def get_score_breakdown(self, resume_text: str, job_description: str = "") -> Dict[str, any]:
        """
        Get detailed score breakdown with explanations
        """
        if job_description.strip():
            scores = self.score_resume_with_job_description(resume_text, job_description)
        else:
            scores = self.score_resume(resume_text)
        
        # Generate feedback based on scores
        feedback = {
            'overall_feedback': self._get_overall_feedback(scores['overallScore']),
            'structure_feedback': self._get_structure_feedback(scores['structureScore']),
            'keyword_feedback': self._get_keyword_feedback(scores['keywordScore']),
            'format_feedback': self._get_format_feedback(scores['formatScore'])
        }
        
        if 'jobMatchScore' in scores:
            feedback['job_match_feedback'] = self._get_job_match_feedback(scores['jobMatchScore'])
        
        return {
            'scores': scores,
            'feedback': feedback,
            'recommendations': self._get_recommendations(scores, job_description)
        }
    
    def _get_overall_feedback(self, score: int) -> str:
        """Generate overall feedback based on score"""
        if score >= 80:
            return "Excellent! Your resume is very ATS-friendly and well-optimized."
        elif score >= 70:
            return "Good! Your resume should perform well with most ATS systems."
        elif score >= 60:
            return "Fair. Some improvements could help your resume perform better."
        elif score >= 50:
            return "Needs improvement. Consider optimizing for ATS compatibility."
        else:
            return "Significant improvements needed for ATS compatibility."
    
    def _get_structure_feedback(self, score: int) -> str:
        """Generate structure feedback"""
        if score >= 80:
            return "Great structure with all essential sections present and properly formatted."
        elif score >= 60:
            return "Good structure, minor improvements possible in section organization."
        else:
            return "Structure needs improvement. Ensure contact info, experience, and skills sections are clearly labeled."
    
    def _get_keyword_feedback(self, score: int) -> str:
        """Generate keyword feedback"""
        if score >= 70:
            return "Strong keyword presence relevant to your industry and role."
        elif score >= 50:
            return "Good keyword usage, consider adding more relevant technical terms and action verbs."
        else:
            return "Keyword optimization needed. Add more industry-relevant terms, skills, and quantifiable achievements."
    
    def _get_format_feedback(self, score: int) -> str:
        """Generate format feedback"""
        if score >= 80:
            return "Excellent formatting that ATS systems can easily parse and process."
        elif score >= 60:
            return "Good formatting with minor areas for improvement in readability and structure."
        else:
            return "Format improvements needed. Focus on clean, simple formatting, quantifiable achievements, and consistent styling."
    
    def _get_job_match_feedback(self, score: int) -> str:
        """Generate job match feedback"""
        if score >= 80:
            return "Excellent match! Your resume aligns very well with the job requirements."
        elif score >= 60:
            return "Good match. Your resume covers most of the key requirements for this role."
        elif score >= 40:
            return "Partial match. Consider highlighting more relevant skills and experiences for this specific role."
        else:
            return "Limited match. Significant optimization needed to align with job requirements."
    
    def _get_recommendations(self, scores: Dict[str, int], job_description: str = "") -> List[str]:
        """Generate specific recommendations based on scores"""
        recommendations = []
        
        if scores['structureScore'] < 70:
            recommendations.append("Add clear section headers like 'Professional Experience', 'Education', and 'Technical Skills'")
            recommendations.append("Include complete contact information with email and phone number")
            recommendations.append("Add employment dates and durations for all work experience entries")
        
        if scores['keywordScore'] < 60:
            recommendations.append("Include more industry-relevant technical skills and keywords")
            recommendations.append("Use strong action verbs like 'developed', 'implemented', 'led' to describe achievements")
            recommendations.append("Add relevant certifications and professional qualifications")
        
        if scores['formatScore'] < 70:
            recommendations.append("Include quantifiable achievements with specific numbers and percentages")
            recommendations.append("Keep formatting simple and avoid special characters or complex layouts")
            recommendations.append("Ensure consistent capitalization and professional language throughout")
        
        if job_description and 'jobMatchScore' in scores and scores['jobMatchScore'] < 60:
            recommendations.append("Tailor your resume more specifically to this job posting")
            recommendations.append("Highlight experiences and skills that directly match the job requirements")
            recommendations.append("Use keywords and terminology from the job description where relevant")
        
        if not recommendations:
            recommendations.append("Excellent work! Your resume is well-optimized for ATS systems and job matching")
        
        return recommendations