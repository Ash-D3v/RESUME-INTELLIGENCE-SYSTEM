# resume_parser.py - Enhanced Resume Parser (using your working parser)
import re
import json
import os
from datetime import datetime
from typing import Dict, List, Any
from collections import Counter

class ResumeParser:
    """Enhanced Resume Parser with comprehensive skill extraction and parsing"""
    
    def __init__(self):
        self.analytics_file = "resume_analytics.json"
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills using the existing working logic from your codebase
        Enhanced with more comprehensive skill patterns
        """
        text = re.sub(r'\s+', ' ', text)
        text_lower = text.lower()
        
        skills = {
            "programming_languages": [],
            "frameworks_libraries": [],
            "tools_technologies": [],
            "databases": [],
            "cloud_platforms": [],
            "methodologies": [],
            "soft_skills": [],
            "certifications": [],
            "web_technologies": [],
            "mobile_technologies": [],
            "data_science": [],
            "devops": []
        }
        
        # Enhanced skill patterns with more comprehensive coverage
        skill_patterns = {
            "programming_languages": [
                r'\b(?:python|java|javascript|js|typescript|ts|c\+\+|cpp|c#|csharp|php|ruby|go|golang|rust|kotlin|swift|scala|r\b|matlab|sql|html|css|perl|bash|shell|powershell|objective-c|dart|lua|haskell|clojure|elixir|erlang)\b'
            ],
            "frameworks_libraries": [
                r'\b(?:react|reactjs|angular|angularjs|vue|vuejs|jquery|bootstrap|tailwind|material-ui|mui|chakra)\b',
                r'\b(?:django|flask|fastapi|spring boot|spring|express|nodejs|next\.js|nuxt|gatsby|svelte)\b',
                r'\b(?:tensorflow|pytorch|pandas|numpy|scikit-learn|sklearn|opencv|matplotlib|seaborn|plotly|keras|scipy)\b',
                r'\b(?:laravel|symfony|codeigniter|rails|sinatra|asp\.net|\.net core|blazor)\b'
            ],
            "web_technologies": [
                r'\b(?:html5|css3|sass|scss|less|webpack|babel|gulp|grunt|npm|yarn|bower)\b',
                r'\b(?:rest|restful|graphql|soap|api|json|xml|ajax|websocket|pwa)\b'
            ],
            "mobile_technologies": [
                r'\b(?:react native|flutter|ionic|xamarin|cordova|phonegap|android|ios|kotlin|swift)\b'
            ],
            "databases": [
                r'\b(?:mysql|postgresql|postgres|mongodb|redis|sqlite|oracle|sql server|elasticsearch|cassandra|dynamodb|firebase|mariadb|neo4j|couchdb|influxdb)\b'
            ],
            "cloud_platforms": [
                r'\b(?:aws|amazon web services|azure|microsoft azure|gcp|google cloud|heroku|digitalocean|linode|vercel|netlify|cloudflare)\b',
                r'\b(?:ec2|s3|lambda|rds|cloudfront|route53|iam|vpc|ecs|eks|fargate)\b'
            ],
            "devops": [
                r'\b(?:docker|kubernetes|k8s|jenkins|gitlab ci|github actions|circleci|travis ci|terraform|ansible|puppet|chef|vagrant)\b',
                r'\b(?:ci/cd|continuous integration|continuous deployment|infrastructure as code|iac)\b'
            ],
            "tools_technologies": [
                r'\b(?:git|github|gitlab|bitbucket|jira|confluence|slack|trello|figma|sketch|photoshop|illustrator)\b',
                r'\b(?:linux|ubuntu|centos|debian|windows|macos|apache|nginx|postman|swagger|insomnia)\b',
                r'\b(?:visual studio|vscode|intellij|eclipse|pycharm|webstorm|sublime|atom|vim|emacs)\b'
            ],
            "data_science": [
                r'\b(?:machine learning|deep learning|neural networks|natural language processing|nlp|computer vision)\b',
                r'\b(?:data analysis|data mining|big data|hadoop|spark|kafka|airflow|jupyter|tableau|power bi)\b',
                r'\b(?:statistics|statistical analysis|predictive modeling|data visualization|etl|data warehousing)\b'
            ],
            "methodologies": [
                r'\b(?:agile|scrum|kanban|devops|tdd|bdd|waterfall|lean|six sigma|design thinking|mvp)\b',
                r'\b(?:test driven development|behavior driven development|continuous integration|pair programming)\b'
            ],
            "soft_skills": [
                r'\b(?:leadership|management|communication|teamwork|problem solving|analytical|creative|innovative|adaptable)\b',
                r'\b(?:project management|team lead|mentor|training|presentation|negotiation|critical thinking|time management)\b',
                r'\b(?:collaboration|cross-functional|stakeholder management|client relations|public speaking)\b'
            ],
            "certifications": [
                r'\b(?:aws certified|azure certified|google cloud certified|pmp|scrum master|cissp|cism|comptia)\b',
                r'\b(?:oracle certified|microsoft certified|cisco certified|vmware certified|salesforce certified)\b',
                r'\b(?:certified kubernetes|ckad|cka|cks|terraform certified|docker certified)\b'
            ]
        }
        
        # Extract skills using patterns
        for category, patterns in skill_patterns.items():
            found_skills = set()
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                found_skills.update(matches)
            skills[category] = list(found_skills)
        
        # Remove duplicates and empty categories
        for category in skills:
            skills[category] = list(set(skills[category]))
        
        skills = {k: v for k, v in skills.items() if v}
        
        return skills
    
    def parse_resume(self, text: str) -> Dict[str, Any]:
        """
        Parse resume text and extract structured information
        """
        parsed_resume = {
            "contact_info": self._extract_contact_info(text),
            "education": self._extract_education(text),
            "experience": self._extract_experience(text),
            "skills": self.extract_skills(text),
            "certifications": self._extract_certifications(text),
            "summary": self._extract_summary(text),
            "projects": self._extract_projects(text)
        }
        
        # Save analytics
        self._save_analytics(parsed_resume)
        
        return parsed_resume
    
    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information"""
        contact_info = {}
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info['email'] = email_match.group()
        
        # Phone extraction
        phone_patterns = [
            r'(\+?\d{1,3}[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}'
        ]
        
        for pattern in phone_patterns:
            phone_match = re.search(pattern, text)
            if phone_match:
                contact_info['phone'] = phone_match.group()
                break
        
        # LinkedIn extraction
        linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/pub/)([A-Za-z0-9\-\.]+)'
        linkedin_match = re.search(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_match:
            contact_info['linkedin'] = f"linkedin.com/in/{linkedin_match.group(1)}"
        
        # GitHub extraction
        github_pattern = r'(?:github\.com/)([A-Za-z0-9\-\.]+)'
        github_match = re.search(github_pattern, text, re.IGNORECASE)
        if github_match:
            contact_info['github'] = f"github.com/{github_match.group(1)}"
        
        return contact_info
    
    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information"""
        education = []
        
        # Common degree patterns
        degree_patterns = [
            r'(?:bachelor|master|phd|doctorate|associate|diploma|certificate).*(?:computer science|engineering|mathematics|physics|business|arts|science)',
            r'(?:b\.?tech|m\.?tech|b\.?sc|m\.?sc|b\.?a|m\.?a|mba|phd)',
            r'(?:degree in|graduated from|university|college|institute).*(?:\d{4}|\d{2})'
        ]
        
        for pattern in degree_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                education.append({
                    'degree': match.group(),
                    'context': text[max(0, match.start()-50):match.end()+50].strip()
                })
        
        return education[:3]  # Limit to 3 entries
    
    def _extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience"""
        experience = []
        
        # Look for experience patterns
        experience_patterns = [
            r'(?:software engineer|developer|analyst|manager|lead|senior|junior).*(?:\d{4}|\d{1,2}\s+(?:year|month))',
            r'(?:worked as|employed as|position as).*(?:\d{4}|\d{1,2}\s+(?:year|month))',
            r'(?:\d{4}\s*-\s*\d{4}|\d{4}\s*-\s*present).*(?:company|organization|firm)'
        ]
        
        for pattern in experience_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                experience.append({
                    'role': match.group(),
                    'context': text[max(0, match.start()-30):match.end()+100].strip()
                })
        
        return experience[:5]  # Limit to 5 entries
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications"""
        cert_patterns = [
            r'(?:certified|certification).*(?:aws|azure|google|microsoft|oracle|cisco)',
            r'(?:pmp|cissp|cism|comptia|scrum master)',
            r'(?:certificate in|certified in).*'
        ]
        
        certifications = []
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            certifications.extend(matches)
        
        return list(set(certifications))[:5]
    
    def _extract_summary(self, text: str) -> str:
        """Extract professional summary"""
        summary_patterns = [
            r'(?:summary|profile|about|objective):\s*(.{100,500})',
            r'(?:professional summary|career objective|personal statement):\s*(.{100,500})'
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: Take first meaningful paragraph
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if len(para) > 100 and any(keyword in para.lower() for keyword in ['experience', 'skilled', 'professional', 'developer', 'engineer']):
                return para.strip()[:500]
        
        return ""
    
    def _extract_projects(self, text: str) -> List[Dict[str, str]]:
        """Extract project information"""
        projects = []
        
        project_patterns = [
            r'(?:project|built|developed|created).*(?:using|with|in).*(?:python|java|react|node|web|mobile)',
            r'(?:github|portfolio|demo).*(?:project|application|website)'
        ]
        
        for pattern in project_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                projects.append({
                    'description': match.group(),
                    'context': text[max(0, match.start()-30):match.end()+100].strip()
                })
        
        return projects[:3]
    
    def _save_analytics(self, parsed_resume: Dict[str, Any]) -> None:
        """Save analytics data"""
        try:
            analytics = {
                'timestamp': datetime.now().isoformat(),
                'skills_count': sum(len(skills) for skills in parsed_resume['skills'].values()),
                'has_contact': bool(parsed_resume['contact_info']),
                'has_education': bool(parsed_resume['education']),
                'has_experience': bool(parsed_resume['experience']),
                'has_projects': bool(parsed_resume['projects'])
            }
            
            if os.path.exists(self.analytics_file):
                with open(self.analytics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data['analytics'].append(analytics)
            else:
                data = {'analytics': [analytics]}
            
            with open(self.analytics_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving analytics: {e}")
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get resume processing analytics"""
        try:
            if os.path.exists(self.analytics_file):
                with open(self.analytics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                analytics_list = data.get('analytics', [])
                
                if not analytics_list:
                    return {"message": "No analytics data available"}
                
                total_resumes = len(analytics_list)
                avg_skills = sum(a['skills_count'] for a in analytics_list) / total_resumes
                
                return {
                    "total_resumes_processed": total_resumes,
                    "average_skills_per_resume": round(avg_skills, 2),
                    "completion_rates": {
                        "contact_info": sum(1 for a in analytics_list if a['has_contact']) / total_resumes,
                        "education": sum(1 for a in analytics_list if a['has_education']) / total_resumes,
                        "experience": sum(1 for a in analytics_list if a['has_experience']) / total_resumes,
                        "projects": sum(1 for a in analytics_list if a['has_projects']) / total_resumes
                    }
                }
            else:
                return {"message": "No analytics data available"}
        except Exception as e:
            return {"error": f"Error retrieving analytics: {str(e)}"}