"""
ML Model Training Script for Google Colab
Train Word2Vec, Job Fit Classifier, and Job Clustering models
"""

import os
import json
import pickle
import requests
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
import logging
import time

# Install required packages
import subprocess
import sys

def install_packages():
    """Install required packages in Colab"""
    packages = [
        "numpy>=1.21.0,<2.0.0",
        "scikit-learn>=1.0.0", 
        "nltk>=3.8.0",
        "gensim>=4.3.0",
        "requests>=2.31.0",
        "datasets>=2.18.0"
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Uncomment the line below if running in Colab for the first time
# install_packages()

# Import ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import gensim
from gensim.models import Word2Vec
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelTrainer:
    """Simplified ML Model Trainer for Colab"""
    
    def __init__(self):
        self.models_dir = "ml_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'])
    
    def fetch_job_data(self):
        """Fetch job data using robust multi-strategy approach.

        1) Try Hugging Face datasets API (recommended).
        2) Fallback to HTTP datasets-server with retries and pagination.
        3) Fallback to a small synthetic dataset to unblock training.
        """
        dataset_id = "muhammetakkurt/naukri-jobs-dataset"

        # Strategy 1: Hugging Face datasets
        try:
            try:
                from datasets import load_dataset  
            except Exception:
                # Attempt to install if missing (Colab-safe)
                subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets>=2.18.0"]) 
                from datasets import load_dataset  # type: ignore

            logger.info("Fetching jobs via Hugging Face datasets.load_dataset()...")
            ds = load_dataset(dataset_id, split="train")
            jobs_data: List[Dict[str, Any]] = []
            for idx, row in enumerate(ds):
                job = {
                    'title': self._clean_text(row.get('Job Title', '')),
                    'company': self._clean_text(row.get('Company', '')),
                    'location': self._clean_text(row.get('Location', '')),
                    'experience': self._clean_text(row.get('Experience', '')),
                    'salary': self._clean_text(row.get('Salary', '')),
                    'skills': self._extract_skills_from_text(row.get('Skills', '')),
                    'description': self._clean_text(row.get('Job Description', '')),
                    'qualification': self._clean_text(row.get('Qualification', '')),
                    'job_id': row.get('Unnamed: 0', idx)
                }
                if job['title'] and job['company']:
                    jobs_data.append(job)

            if jobs_data:
                logger.info(f"Successfully fetched {len(jobs_data)} jobs via datasets API")
                return jobs_data
        except Exception as e:
            logger.warning(f"datasets API failed, will try HTTP fallback: {e}")

        # Strategy 2: HTTP datasets-server with retries & pagination
        try:
            logger.info("Fetching jobs via HTTP datasets-server with retries...")
            base_url = "https://datasets-server.huggingface.co/rows"
            params = {
                "dataset": dataset_id,
                "config": "default",
                "split": "train",
                "offset": 0,
                "length": 1000,
            }
            jobs_data: List[Dict[str, Any]] = []

            while True:
                success = False
                last_err = None
                for attempt in range(3):
                    try:
                        resp = requests.get(base_url, params=params, timeout=30)
                        resp.raise_for_status()
                        data = resp.json()
                        rows = data.get('rows', [])
                        for row in rows:
                            row_data = row.get('row', {})
                            job = {
                                'title': self._clean_text(row_data.get('Job Title', '')),
                                'company': self._clean_text(row_data.get('Company', '')),
                                'location': self._clean_text(row_data.get('Location', '')),
                                'experience': self._clean_text(row_data.get('Experience', '')),
                                'salary': self._clean_text(row_data.get('Salary', '')),
                                'skills': self._extract_skills_from_text(row_data.get('Skills', '')),
                                'description': self._clean_text(row_data.get('Job Description', '')),
                                'qualification': self._clean_text(row_data.get('Qualification', '')),
                                'job_id': row_data.get('Unnamed: 0', len(jobs_data))
                            }
                            if job['title'] and job['company']:
                                jobs_data.append(job)
                        success = True
                        break
                    except Exception as err:
                        last_err = err
                        # exponential backoff
                        time.sleep(1.5 * (attempt + 1))

                if not success:
                    logger.warning(f"HTTP page fetch failed after retries: {last_err}")
                    break

                # Stop if fewer than requested rows returned (no more pages)
                if len(rows) < params["length"]:
                    break
                params["offset"] += params["length"]

            if jobs_data:
                logger.info(f"Successfully fetched {len(jobs_data)} jobs via HTTP fallback")
                return jobs_data
        except Exception as e:
            logger.warning(f"HTTP fallback failed, will use synthetic data: {e}")

        # Strategy 3: Generate comprehensive synthetic dataset
        logger.info("Generating comprehensive synthetic job dataset (10,000+ jobs)")
        return self._generate_large_synthetic_dataset()
    
    def _generate_large_synthetic_dataset(self, n_jobs: int = 10000) -> List[Dict[str, Any]]:
        """Generate a large synthetic dataset for robust model training"""
        import random
        
        # Comprehensive skill sets by domain
        skill_domains = {
            'programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'scala', 'kotlin', 'swift', 'dart', 'r'],
            'web_frontend': ['html', 'css', 'javascript', 'react', 'angular', 'vue.js', 'svelte', 'next.js', 'nuxt.js', 'webpack', 'sass', 'less', 'tailwind'],
            'web_backend': ['node.js', 'express', 'django', 'flask', 'fastapi', 'spring', 'spring boot', 'laravel', 'rails', 'asp.net', 'gin', 'fiber'],
            'mobile': ['react native', 'flutter', 'android', 'ios', 'swift', 'kotlin', 'xamarin', 'ionic', 'cordova'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'matplotlib', 'seaborn', 'plotly', 'jupyter', 'r', 'stata'],
            'ml_ai': ['machine learning', 'deep learning', 'neural networks', 'nlp', 'computer vision', 'reinforcement learning', 'mlops', 'model deployment'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'elasticsearch', 'neo4j', 'dynamodb', 'sqlite'],
            'cloud_aws': ['aws', 'ec2', 's3', 'lambda', 'rds', 'cloudformation', 'eks', 'ecs', 'api gateway', 'cloudwatch'],
            'cloud_azure': ['azure', 'azure functions', 'cosmos db', 'azure sql', 'azure devops', 'aks', 'azure storage'],
            'cloud_gcp': ['gcp', 'google cloud', 'bigquery', 'cloud functions', 'gke', 'cloud storage', 'firebase'],
            'devops': ['docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions', 'terraform', 'ansible', 'chef', 'puppet'],
            'tools': ['git', 'jira', 'confluence', 'slack', 'postman', 'swagger', 'linux', 'bash', 'powershell', 'vim', 'vscode'],
            'testing': ['unit testing', 'integration testing', 'selenium', 'cypress', 'jest', 'pytest', 'junit', 'testng'],
            'security': ['oauth', 'jwt', 'ssl', 'encryption', 'penetration testing', 'vulnerability assessment', 'owasp'],
            'analytics': ['google analytics', 'tableau', 'power bi', 'looker', 'qlik', 'excel', 'spark', 'hadoop'],
            'design': ['figma', 'sketch', 'adobe xd', 'photoshop', 'illustrator', 'ui/ux', 'wireframing', 'prototyping']
        }
        
        # Job templates with realistic descriptions
        job_templates = [
            {
                'title': 'Software Engineer',
                'domains': ['programming', 'web_backend', 'databases', 'tools'],
                'description': 'Design and develop scalable software applications. Work with cross-functional teams to deliver high-quality code. Participate in code reviews and maintain technical documentation.',
                'companies': ['TechCorp', 'InnovateSoft', 'CodeFactory', 'DevSolutions', 'ByteWorks']
            },
            {
                'title': 'Data Scientist',
                'domains': ['data_science', 'ml_ai', 'programming', 'analytics'],
                'description': 'Analyze complex datasets to extract actionable insights. Build predictive models and machine learning pipelines. Present findings to stakeholders and drive data-driven decisions.',
                'companies': ['DataLabs', 'Analytics Pro', 'AI Insights', 'MetricsCorp', 'DataDriven']
            },
            {
                'title': 'Frontend Developer',
                'domains': ['web_frontend', 'programming', 'design', 'tools'],
                'description': 'Create responsive and interactive user interfaces. Collaborate with designers to implement pixel-perfect designs. Optimize applications for maximum speed and scalability.',
                'companies': ['WebStudio', 'UI Masters', 'Frontend Labs', 'DesignTech', 'UserFirst']
            },
            {
                'title': 'DevOps Engineer',
                'domains': ['devops', 'cloud_aws', 'tools', 'security'],
                'description': 'Manage CI/CD pipelines and cloud infrastructure. Automate deployment processes and monitor system performance. Ensure security and compliance across all environments.',
                'companies': ['CloudOps', 'InfraTech', 'DevSecure', 'AutoDeploy', 'ScaleWorks']
            },
            {
                'title': 'Full Stack Developer',
                'domains': ['web_frontend', 'web_backend', 'databases', 'programming'],
                'description': 'Develop end-to-end web applications from frontend to backend. Design database schemas and implement RESTful APIs. Ensure seamless integration between all system components.',
                'companies': ['FullStack Inc', 'WebComplete', 'EndToEnd Tech', 'TotalDev', 'AllStack']
            },
            {
                'title': 'Machine Learning Engineer',
                'domains': ['ml_ai', 'data_science', 'cloud_aws', 'programming'],
                'description': 'Deploy machine learning models into production systems. Build MLOps pipelines for model training and monitoring. Optimize model performance and scalability.',
                'companies': ['MLOps Co', 'AI Deploy', 'ModelWorks', 'ML Systems', 'AutoML']
            },
            {
                'title': 'Mobile Developer',
                'domains': ['mobile', 'programming', 'design', 'tools'],
                'description': 'Develop native and cross-platform mobile applications. Implement user-friendly interfaces and smooth user experiences. Integrate with backend APIs and third-party services.',
                'companies': ['MobileFirst', 'AppCraft', 'Mobile Solutions', 'TouchTech', 'AppMasters']
            },
            {
                'title': 'Database Administrator',
                'domains': ['databases', 'cloud_aws', 'security', 'tools'],
                'description': 'Manage and optimize database systems for performance and reliability. Implement backup and recovery procedures. Ensure data security and compliance with regulations.',
                'companies': ['DataSecure', 'DB Masters', 'InfoSafe', 'DataGuard', 'DB Solutions']
            },
            {
                'title': 'Cloud Architect',
                'domains': ['cloud_aws', 'cloud_azure', 'devops', 'security'],
                'description': 'Design scalable and secure cloud architectures. Lead cloud migration projects and optimize cloud costs. Implement best practices for cloud security and governance.',
                'companies': ['CloudArch', 'SkyTech', 'CloudMasters', 'ArcSolutions', 'CloudPro']
            },
            {
                'title': 'QA Engineer',
                'domains': ['testing', 'tools', 'programming', 'web_frontend'],
                'description': 'Design and execute comprehensive test plans. Automate testing processes and maintain test frameworks. Collaborate with development teams to ensure quality deliverables.',
                'companies': ['QualityFirst', 'TestPro', 'QA Masters', 'BugFree', 'TestWorks']
            }
        ]
        
        locations = ['Bengaluru', 'Mumbai', 'Delhi', 'Hyderabad', 'Pune', 'Chennai', 'Kolkata', 'Ahmedabad', 'Remote', 'Gurgaon']
        experience_levels = ['0-1 years', '1-3 years', '2-4 years', '3-5 years', '4-7 years', '5-8 years', '6-10 years']
        salary_ranges = ['3-6 LPA', '5-10 LPA', '8-15 LPA', '12-20 LPA', '15-25 LPA', '20-30 LPA', '25-40 LPA']
        qualifications = ['B.Tech/BE', 'MCA', 'M.Tech', 'Any Graduate', 'B.Sc Computer Science', 'BCA']
        
        synthetic_jobs = []
        
        for i in range(n_jobs):
            template = random.choice(job_templates)
            
            # Select skills from relevant domains
            job_skills = []
            for domain in template['domains']:
                domain_skills = skill_domains.get(domain, [])
                # Pick 2-5 skills from each domain
                selected_skills = random.sample(domain_skills, min(random.randint(2, 5), len(domain_skills)))
                job_skills.extend(selected_skills)
            
            # Add some random skills from other domains (10% chance each)
            for domain, skills in skill_domains.items():
                if domain not in template['domains'] and random.random() < 0.1:
                    job_skills.append(random.choice(skills))
            
            # Remove duplicates and limit to reasonable number
            job_skills = list(set(job_skills))[:15]
            
            job = {
                'title': template['title'],
                'company': random.choice(template['companies']),
                'location': random.choice(locations),
                'experience': random.choice(experience_levels),
                'salary': random.choice(salary_ranges),
                'skills': job_skills,
                'description': template['description'],
                'qualification': random.choice(qualifications),
                'job_id': i + 1
            }
            
            synthetic_jobs.append(job)
        
        logger.info(f"Generated {len(synthetic_jobs)} synthetic jobs with comprehensive skill coverage")
        return synthetic_jobs
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        if not text or text == 'nan':
            return ""
        return str(text).strip()
    
    def _extract_skills_from_text(self, skills_text):
        """Extract skills from skills text"""
        if not skills_text or skills_text == 'nan':
            return []
        
        skills_text = str(skills_text).lower()
        separators = [',', '|', ';', '/', '&', 'and', '\n', '\t']
        
        skills = [skills_text]
        for sep in separators:
            new_skills = []
            for skill in skills:
                new_skills.extend([s.strip() for s in skill.split(sep) if s.strip()])
            skills = new_skills
        
        filtered_skills = []
        stop_words = {'and', 'or', 'the', 'in', 'at', 'to', 'for', 'of', 'with', 'on', 'by'}
        
        for skill in skills:
            skill = skill.strip()
            if len(skill) >= 2 and skill not in stop_words:
                filtered_skills.append(skill)
        
        return list(set(filtered_skills))[:20]
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for Word2Vec"""
        if not text:
            return []
        
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        return tokens
    
    def train_word2vec_model(self, jobs_data: List[Dict]):
        """Train Word2Vec model on job skills and descriptions"""
        logger.info("Training Word2Vec model...")
        
        # Generate comprehensive text corpus for Word2Vec
        texts = []
        
        # Extended skill-related sentences for better vocabulary
        skill_contexts = [
            "experience with", "proficient in", "skilled at", "expertise in", "knowledge of",
            "working with", "developing using", "implementing with", "building applications using",
            "creating solutions with", "designing systems using", "maintaining code in",
            "debugging applications built with", "optimizing performance using", "deploying with"
        ]
        
        for job in jobs_data:
            # Add individual skills as sentences
            if job.get('skills'):
                for skill in job['skills']:
                    texts.append(skill)
                    # Add skill in various contexts
                    for context in skill_contexts[:3]:  # Use first 3 contexts
                        texts.append(f"{context} {skill}")
                
                # Add combined skills
                skill_text = ' '.join(job['skills'])
                texts.append(skill_text)
                
                # Add job title with skills
                if job.get('title'):
                    texts.append(f"{job['title']} {skill_text}")
            
            # Add full descriptions
            if job.get('description'):
                texts.append(job['description'])
                
                # Add description with skills mixed in
                if job.get('skills'):
                    desc_with_skills = f"{job['description']} requires {' '.join(job['skills'][:5])}"
                    texts.append(desc_with_skills)
            
            # Add job title variations
            if job.get('title'):
                texts.append(job['title'])
                if job.get('location'):
                    texts.append(f"{job['title']} in {job['location']}")
                if job.get('experience'):
                    texts.append(f"{job['title']} with {job['experience']} experience")
        
        # Add domain-specific sentences to expand vocabulary
        domain_sentences = [
            "python programming language for data science and web development",
            "javascript frameworks like react angular vue for frontend development",
            "machine learning algorithms using scikit learn tensorflow pytorch",
            "cloud computing platforms aws azure gcp for scalable applications",
            "database management systems mysql postgresql mongodb for data storage",
            "devops tools docker kubernetes jenkins for continuous integration deployment",
            "web development technologies html css javascript for user interfaces",
            "backend development using node.js django flask spring boot frameworks",
            "mobile app development react native flutter android ios platforms",
            "data analysis visualization pandas numpy matplotlib seaborn libraries",
            "software testing automation selenium cypress jest pytest frameworks",
            "version control git github gitlab for collaborative development",
            "agile methodologies scrum kanban for project management",
            "api development rest graphql microservices architecture patterns",
            "security practices oauth jwt ssl encryption for application protection"
        ]
        
        texts.extend(domain_sentences)
        
        # Preprocess texts into sentences
        sentences = []
        for text in texts:
            if text and len(text.strip()) > 0:
                processed = self.preprocess_text(text)
                if len(processed) > 1:  # Only include sentences with multiple words
                    sentences.append(processed)
        
        logger.info(f"Prepared {len(sentences)} sentences for Word2Vec training")
        
        if len(sentences) < 50:
            logger.warning("Not enough data for Word2Vec training")
            return None
        
        # Train Word2Vec model with much larger parameters for ~50MB model
        word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=400,  # Much larger vector size (200 -> 400)
            window=12,        # Larger context window (8 -> 12)
            min_count=1,      # Include all words for maximum vocabulary
            workers=4,
            epochs=50,        # More training epochs (20 -> 50)
            sg=1,            # Skip-gram model
            negative=20,     # More negative sampling (10 -> 20)
            alpha=0.025,     # Learning rate
            min_alpha=0.0001 # Minimum learning rate
        )
        
        # Save model
        model_path = os.path.join(self.models_dir, "word2vec_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(word2vec_model, f)
        
        logger.info(f"Word2Vec model trained with vocabulary size: {len(word2vec_model.wv)}")
        logger.info(f"Model vector size: {word2vec_model.vector_size}")
        return word2vec_model
    
    def extract_skill_features(self, skills: List[str]) -> Dict[str, float]:
        """Extract features from skills for ML models"""
        features = {}
        
        skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust'],
            'web_dev': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'keras', 'pytorch', 'matplotlib'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'devops': ['jenkins', 'git', 'ci/cd', 'linux', 'bash', 'ansible']
        }
        
        skills_lower = [skill.lower() for skill in skills]
        
        for category, category_skills in skill_categories.items():
            count = sum(1 for skill in skills_lower if any(cat_skill in skill for cat_skill in category_skills))
            features[f'{category}_count'] = count
        
        features['total_skills'] = len(skills)
        categories_present = sum(1 for category in skill_categories.keys() if features[f'{category}_count'] > 0)
        features['skill_diversity'] = categories_present
        
        return features
    
    def generate_synthetic_training_data(self, jobs_data: List[Dict], n_samples: int = 5000):
        """Generate synthetic training data for job fit classifier"""
        logger.info(f"Generating {n_samples} synthetic training samples...")
        
        training_data = []
        
        for i in range(n_samples):
            # Pick a random job
            job = np.random.choice(jobs_data)
            job_skills = job.get('skills', [])
            
            if not job_skills:
                continue
            
            # Generate user skills with varying overlap
            overlap_ratio = np.random.uniform(0.1, 0.9)
            n_overlap = max(1, int(len(job_skills) * overlap_ratio))
            
            # Select overlapping skills
            overlapping_skills = np.random.choice(job_skills, size=min(n_overlap, len(job_skills)), replace=False).tolist()
            
            # Add some random skills from other jobs
            all_skills = []
            for other_job in np.random.choice(jobs_data, size=5, replace=True):
                all_skills.extend(other_job.get('skills', []))
            
            additional_skills = []
            if all_skills:
                n_additional = np.random.randint(0, 5)
                additional_skills = np.random.choice(list(set(all_skills)), size=min(n_additional, len(set(all_skills))), replace=False).tolist()
            
            user_skills = overlapping_skills + additional_skills
            
            # Calculate fit score based on overlap
            fit_score = int(overlap_ratio * 100)
            
            training_data.append({
                'user_skills': user_skills,
                'job_skills': job_skills,
                'fit_score': fit_score
            })
        
        return training_data
    
    def train_job_fit_classifier(self, training_data: List[Dict]):
        """Train job fit classifier"""
        logger.info("Training job fit classifier...")
        
        if len(training_data) < 50:
            logger.warning("Not enough training data for classification model")
            return None, None
        
        # Prepare features and labels
        X = []
        y = []
        
        for data in training_data:
            user_skills = data.get('user_skills', [])
            job_skills = data.get('job_skills', [])
            fit_score = data.get('fit_score', 0)
            
            # Extract features
            user_features = self.extract_skill_features(user_skills)
            job_features = self.extract_skill_features(job_skills)
            
            # Combine features
            combined_features = []
            for category in ['programming', 'web_dev', 'data_science', 'databases', 'cloud', 'devops']:
                combined_features.extend([
                    user_features.get(f'{category}_count', 0),
                    job_features.get(f'{category}_count', 0),
                    user_features.get(f'{category}_count', 0) - job_features.get(f'{category}_count', 0)
                ])
            
            combined_features.extend([
                user_features.get('total_skills', 0),
                job_features.get('total_skills', 0),
                user_features.get('skill_diversity', 0),
                job_features.get('skill_diversity', 0)
            ])
            
            X.append(combined_features)
            y.append(1 if fit_score >= 70 else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train classifier
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)
        
        logger.info(f"Classifier trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        
        # Save models
        classifier_path = os.path.join(self.models_dir, "job_fit_classifier.pkl")
        with open(classifier_path, 'wb') as f:
            pickle.dump(classifier, f)
        
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        return classifier, scaler
    
    def train_job_clusters(self, jobs_data: List[Dict], n_clusters: int = 5):
        """Train job clustering model"""
        logger.info(f"Training job clustering with {n_clusters} clusters...")
        
        if len(jobs_data) < n_clusters:
            logger.warning("Not enough job data for clustering")
            return None
        
        # Extract features
        features = []
        for job in jobs_data:
            skills = job.get('skills', [])
            job_features = self.extract_skill_features(skills)
            
            feature_vector = [
                job_features.get('programming_count', 0),
                job_features.get('web_dev_count', 0),
                job_features.get('data_science_count', 0),
                job_features.get('databases_count', 0),
                job_features.get('cloud_count', 0),
                job_features.get('devops_count', 0),
                job_features.get('total_skills', 0),
                job_features.get('skill_diversity', 0)
            ]
            features.append(feature_vector)
        
        # Perform clustering
        X = np.array(features)
        X_scaled = StandardScaler().fit_transform(X)
        
        job_clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = job_clusters.fit_predict(X_scaled)
        
        # Save clustering model
        clusters_path = os.path.join(self.models_dir, "job_clusters.pkl")
        with open(clusters_path, 'wb') as f:
            pickle.dump(job_clusters, f)
        
        logger.info(f"Job clustering completed with {n_clusters} clusters")
        return job_clusters

def main():
    """Main training function"""
    print("🚀 Starting ML Model Training for Resume Analysis System")
    print("=" * 60)
    
    trainer = MLModelTrainer()
    
    # Step 1: Fetch job data
    print("\n📊 Step 1: Fetching job data...")
    jobs_data = trainer.fetch_job_data()
    
    if not jobs_data:
        print("❌ Failed to fetch job data. Exiting.")
        return
    
    print(f"✅ Successfully loaded {len(jobs_data)} jobs")
    
    # Step 2: Train Word2Vec model
    print("\n🧠 Step 2: Training Word2Vec model...")
    word2vec_model = trainer.train_word2vec_model(jobs_data)
    
    if word2vec_model:
        print("✅ Word2Vec model trained and saved")
    else:
        print("❌ Word2Vec training failed")
    
    # Step 3: Generate training data and train classifier
    print("\n🎯 Step 3: Training job fit classifier...")
    training_data = trainer.generate_synthetic_training_data(jobs_data, n_samples=2000)
    classifier, scaler = trainer.train_job_fit_classifier(training_data)
    
    if classifier and scaler:
        print("✅ Job fit classifier trained and saved")
    else:
        print("❌ Classifier training failed")
    
    # Step 4: Train job clustering
    print("\n🔍 Step 4: Training job clustering...")
    job_clusters = trainer.train_job_clusters(jobs_data, n_clusters=8)
    
    if job_clusters:
        print("✅ Job clustering model trained and saved")
    else:
        print("❌ Clustering training failed")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print(f"📁 Models saved in: {trainer.models_dir}/")
    
    # List saved files
    model_files = os.listdir(trainer.models_dir)
    if model_files:
        print("\n📋 Saved model files:")
        for file in model_files:
            file_path = os.path.join(trainer.models_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  • {file} ({size_mb:.2f} MB)")
    
    print("\n💡 Next steps:")
    print("1. Download all files from the ml_models/ folder")
    print("2. Place them in your local project's ml_models/ directory")
    print("3. Run your application - models will be loaded automatically!")

if __name__ == "__main__":
    main()
