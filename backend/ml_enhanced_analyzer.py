import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import json
import os
from typing import List, Dict, Any
from collections import defaultdict
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class MLEnhancedAnalyzer:
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        self.word2vec_model = None
        self.job_fit_classifier = None
        self.job_clusters = None
        self.scaler = StandardScaler()
        
        self.models_dir = "ml_models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'])
        
        self._load_models()
    
    def preprocess_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        return tokens
    
    def train_word2vec_model(self, texts: List[str], vector_size: int = 100):
        logger.info("Training Word2Vec model...")
        
        sentences = [self.preprocess_text(text) for text in texts]
        sentences = [sent for sent in sentences if len(sent) > 0]
        
        if len(sentences) < 10:
            logger.warning("Not enough data for Word2Vec training. Using fallback.")
            return
        
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=5,
            min_count=2,
            workers=4,
            epochs=10
        )
        
        model_path = os.path.join(self.models_dir, "word2vec_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.word2vec_model, f)
        
        logger.info(f"Word2Vec model trained with vocabulary size: {len(self.word2vec_model.wv)}")
    
    def get_semantic_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate semantic similarity between skills using Word2Vec"""
        if not self.word2vec_model:
            return 0.0
        
        try:
            skill1_clean = skill1.lower().replace(' ', '_')
            skill2_clean = skill2.lower().replace(' ', '_')
            
            if skill1_clean in self.word2vec_model.wv and skill2_clean in self.word2vec_model.wv:
                return self.word2vec_model.wv.similarity(skill1_clean, skill2_clean)
            else:
                return 0.5 if skill1_clean in skill2_clean or skill2_clean in skill1_clean else 0.0
        except:
            return 0.0
    
    def calculate_tfidf_similarity(self, resume_text: str, job_description: str) -> float:
        """Calculate TF-IDF cosine similarity between resume and job description"""
        try:
            documents = [resume_text, job_description]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating TF-IDF similarity: {e}")
            return 0.0
    
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
        
        # Total skills
        features['total_skills'] = len(skills)
        
        # Skill diversity (unique categories)
        categories_present = sum(1 for category in skill_categories.keys() if features[f'{category}_count'] > 0)
        features['skill_diversity'] = categories_present
        
        return features
    
    def train_job_fit_classifier(self, training_data: List[Dict]):
        """Train classification model to predict job fit probability"""
        logger.info("Training job fit classifier...")
        
        if len(training_data) < 50:
            logger.warning("Not enough training data for classification model")
            return
        
        # Prepare features and labels
        X = []
        y = []
        
        for data in training_data:
            user_skills = data.get('user_skills', [])
            job_skills = data.get('job_skills', [])
            fit_score = data.get('fit_score', 0)  # 0-100 score
            
            # Extract features
            user_features = self.extract_skill_features(user_skills)
            job_features = self.extract_skill_features(job_skills)
            
            # Combine features
            combined_features = []
            for category in ['programming', 'web_dev', 'data_science', 'databases', 'cloud', 'devops']:
                combined_features.extend([
                    user_features.get(f'{category}_count', 0),
                    job_features.get(f'{category}_count', 0),
                    user_features.get(f'{category}_count', 0) - job_features.get(f'{category}_count', 0)  # Gap
                ])
            
            # Add overall metrics
            combined_features.extend([
                user_features.get('total_skills', 0),
                job_features.get('total_skills', 0),
                user_features.get('skill_diversity', 0),
                job_features.get('skill_diversity', 0)
            ])
            
            X.append(combined_features)
            y.append(1 if fit_score >= 70 else 0)  # Binary classification: good fit or not
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest classifier
        self.job_fit_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train model
        self.job_fit_classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.job_fit_classifier.score(X_train, y_train)
        test_score = self.job_fit_classifier.score(X_test, y_test)
        
        logger.info(f"Job fit classifier trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
        
        # Save model
        model_path = os.path.join(self.models_dir, "job_fit_classifier.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.job_fit_classifier, f)
        
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def predict_job_fit_probability(self, user_skills: List[str], job_skills: List[str]) -> float:
        """Predict job fit probability using trained classifier"""
        if not self.job_fit_classifier:
            return 0.5  # Default probability
        
        try:
            # Extract features
            user_features = self.extract_skill_features(user_skills)
            job_features = self.extract_skill_features(job_skills)
            
            # Combine features (same as training)
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
            
            # Scale and predict
            X = np.array([combined_features])
            X_scaled = self.scaler.transform(X)
            
            probability = self.job_fit_classifier.predict_proba(X_scaled)[0][1]  # Probability of good fit
            return float(probability)
            
        except Exception as e:
            logger.error(f"Error predicting job fit: {e}")
            return 0.5
    
    def cluster_job_profiles(self, job_data: List[Dict], n_clusters: int = 5):
        """Cluster job profiles using K-means"""
        logger.info(f"Clustering job profiles into {n_clusters} clusters...")
        
        if len(job_data) < n_clusters:
            logger.warning("Not enough job data for clustering")
            return
        
        # Extract features for clustering
        features = []
        job_titles = []
        
        for job in job_data:
            skills = job.get('skills', [])
            title = job.get('title', 'Unknown')
            
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
            job_titles.append(title)
        
        # Perform clustering
        X = np.array(features)
        X_scaled = StandardScaler().fit_transform(X)
        
        self.job_clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.job_clusters.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_jobs = [job_titles[j] for j, label in enumerate(cluster_labels) if label == i]
            cluster_features = X_scaled[cluster_labels == i]
            
            cluster_analysis[f'cluster_{i}'] = {
                'size': len(cluster_jobs),
                'sample_jobs': cluster_jobs[:5],
                'centroid': cluster_features.mean(axis=0).tolist()
            }
        
        logger.info(f"Job clustering completed: {cluster_analysis}")
        
        # Save clustering model
        model_path = os.path.join(self.models_dir, "job_clusters.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.job_clusters, f)
        
        return cluster_analysis
    
    def collaborative_filtering_recommendations(self, user_skills: List[str], 
                                              user_job_interactions: List[Dict],
                                              all_jobs: List[Dict], 
                                              n_recommendations: int = 5) -> List[Dict]:
        """Generate recommendations using collaborative filtering"""
        logger.info("Generating collaborative filtering recommendations...")
        
        try:
            # Create user-job interaction matrix
            job_ids = [job.get('job_id', i) for i, job in enumerate(all_jobs)]
            user_interactions = defaultdict(float)
            
            # Process user interactions (views, applications, etc.)
            for interaction in user_job_interactions:
                job_id = interaction.get('job_id')
                interaction_type = interaction.get('type', 'view')  # view, apply, save
                
                # Weight different interaction types
                weights = {'view': 1.0, 'save': 2.0, 'apply': 3.0}
                weight = weights.get(interaction_type, 1.0)
                
                if job_id in job_ids:
                    user_interactions[job_id] += weight
            
            # Find similar jobs using content-based similarity
            user_skill_features = self.extract_skill_features(user_skills)
            job_similarities = []
            
            for job in all_jobs:
                job_skills = job.get('skills', [])
                job_features = self.extract_skill_features(job_skills)
                
                # Calculate feature similarity
                similarity = 0.0
                total_features = 0
                
                for feature_name in user_skill_features:
                    if feature_name in job_features:
                        user_val = user_skill_features[feature_name]
                        job_val = job_features[feature_name]
                        
                        if user_val > 0 or job_val > 0:
                            similarity += min(user_val, job_val) / max(user_val, job_val, 1)
                            total_features += 1
                
                if total_features > 0:
                    similarity /= total_features
                
                job_similarities.append({
                    'job': job,
                    'similarity': similarity,
                    'interaction_score': user_interactions.get(job.get('job_id'), 0)
                })
            
            # Combine content similarity with interaction history
            for item in job_similarities:
                # Boost score for jobs user has interacted with
                interaction_boost = min(item['interaction_score'] * 0.1, 0.3)
                item['final_score'] = item['similarity'] + interaction_boost
            
            # Sort by final score and return top recommendations
            job_similarities.sort(key=lambda x: x['final_score'], reverse=True)
            
            recommendations = []
            for item in job_similarities[:n_recommendations]:
                job = item['job']
                recommendations.append({
                    'job_id': job.get('job_id'),
                    'title': job.get('title'),
                    'company': job.get('company'),
                    'similarity_score': round(item['similarity'], 3),
                    'recommendation_score': round(item['final_score'], 3),
                    'skills': job.get('skills', [])[:5]  # Top 5 skills
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {e}")
            return []
    
    def enhanced_skill_matching(self, user_skills: List[str], job_skills: List[str]) -> Dict[str, Any]:
        """Enhanced skill matching using multiple ML techniques"""
        results = {
            'exact_matches': [],
            'semantic_matches': [],
            'tfidf_similarity': 0.0,
            'ml_job_fit_probability': 0.0,
            'overall_match_score': 0.0
        }
        
        try:
            # Exact matches
            user_skills_lower = [skill.lower() for skill in user_skills]
            job_skills_lower = [skill.lower() for skill in job_skills]
            exact_matches = list(set(user_skills_lower) & set(job_skills_lower))
            results['exact_matches'] = exact_matches
            
            # Semantic matches using Word2Vec
            semantic_matches = []
            if self.word2vec_model:
                for user_skill in user_skills:
                    for job_skill in job_skills:
                        similarity = self.get_semantic_similarity(user_skill, job_skill)
                        if similarity > 0.6 and user_skill.lower() not in exact_matches:
                            semantic_matches.append({
                                'user_skill': user_skill,
                                'job_skill': job_skill,
                                'similarity': round(similarity, 3)
                            })
            
            results['semantic_matches'] = semantic_matches
            
            # TF-IDF similarity
            user_text = ' '.join(user_skills)
            job_text = ' '.join(job_skills)
            results['tfidf_similarity'] = self.calculate_tfidf_similarity(user_text, job_text)
            
            # ML job fit probability
            results['ml_job_fit_probability'] = self.predict_job_fit_probability(user_skills, job_skills)
            
            # Calculate overall match score
            exact_score = len(exact_matches) / max(len(job_skills), 1) * 100
            semantic_score = len(semantic_matches) / max(len(job_skills), 1) * 50
            tfidf_score = results['tfidf_similarity'] * 100
            ml_score = results['ml_job_fit_probability'] * 100
            
            # Weighted combination
            overall_score = (
                exact_score * 0.4 +
                semantic_score * 0.2 +
                tfidf_score * 0.2 +
                ml_score * 0.2
            )
            
            results['overall_match_score'] = round(min(overall_score, 100), 2)
            
        except Exception as e:
            logger.error(f"Error in enhanced skill matching: {e}")
        
        return results
    
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            # Load Word2Vec model
            word2vec_path = os.path.join(self.models_dir, "word2vec_model.pkl")
            if os.path.exists(word2vec_path):
                with open(word2vec_path, 'rb') as f:
                    self.word2vec_model = pickle.load(f)
                logger.info("Word2Vec model loaded")
            
            # Load job fit classifier
            classifier_path = os.path.join(self.models_dir, "job_fit_classifier.pkl")
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            
            if os.path.exists(classifier_path) and os.path.exists(scaler_path):
                with open(classifier_path, 'rb') as f:
                    self.job_fit_classifier = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Job fit classifier loaded")
            
            # Load clustering model
            clusters_path = os.path.join(self.models_dir, "job_clusters.pkl")
            if os.path.exists(clusters_path):
                with open(clusters_path, 'rb') as f:
                    self.job_clusters = pickle.load(f)
                logger.info("Job clustering model loaded")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def save_models(self):
        """Save all trained models"""
        try:
            if self.word2vec_model:
                model_path = os.path.join(self.models_dir, "word2vec_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(self.word2vec_model, f)
            
            if self.job_fit_classifier:
                model_path = os.path.join(self.models_dir, "job_fit_classifier.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(self.job_fit_classifier, f)
                
                scaler_path = os.path.join(self.models_dir, "scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            if self.job_clusters:
                model_path = os.path.join(self.models_dir, "job_clusters.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(self.job_clusters, f)
            
            logger.info("All models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")


