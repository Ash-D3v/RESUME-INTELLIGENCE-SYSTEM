# skill_analyzer.py - Skill Gap Analysis Module
import json
import os
from typing import List, Dict, Any, Set
from collections import Counter
from datetime import datetime
import re

class SkillAnalyzer:
    """
    Analyze skill gaps between user skills and job requirements
    Provide insights and prioritization for skill development
    """
    
    def __init__(self):
        self.skill_trends_file = "skill_trends.json"
        
        # Skill categories with their relationships and priorities
        self.skill_categories = {
            'programming_languages': {
                'priority': 'high',
                'related_skills': {
                    'python': ['django', 'flask', 'fastapi', 'pandas', 'numpy'],
                    'javascript': ['react', 'node.js', 'vue', 'angular', 'express'],
                    'java': ['spring', 'hibernate', 'maven', 'gradle'],
                    'c#': ['.net', 'asp.net', 'entity framework'],
                    'php': ['laravel', 'symfony', 'codeigniter'],
                    'ruby': ['rails', 'sinatra']
                }
            },
            'frameworks_libraries': {
                'priority': 'high',
                'dependencies': {
                    'react': 'javascript',
                    'django': 'python',
                    'spring': 'java',
                    'angular': 'typescript',
                    'vue': 'javascript'
                }
            },
            'databases': {
                'priority': 'medium',
                'categories': {
                    'relational': ['mysql', 'postgresql', 'sql server', 'oracle'],
                    'nosql': ['mongodb', 'redis', 'cassandra', 'dynamodb'],
                    'cloud': ['aws rds', 'azure sql', 'cloud sql']
                }
            },
            'cloud_platforms': {
                'priority': 'high',
                'learning_path': ['aws', 'azure', 'gcp'],
                'services': {
                    'aws': ['ec2', 's3', 'lambda', 'rds', 'cloudfront'],
                    'azure': ['virtual machines', 'blob storage', 'functions', 'sql database'],
                    'gcp': ['compute engine', 'cloud storage', 'cloud functions', 'cloud sql']
                }
            },
            'devops': {
                'priority': 'high',
                'learning_sequence': ['git', 'docker', 'kubernetes', 'jenkins', 'terraform']
            },
            'soft_skills': {
                'priority': 'medium',
                'categories': {
                    'leadership': ['team management', 'project management', 'mentoring'],
                    'communication': ['presentation', 'documentation', 'stakeholder management'],
                    'problem_solving': ['analytical thinking', 'debugging', 'troubleshooting']
                }
            }
        }
        
        # Industry demand weights (higher = more in demand)
        self.industry_demand = {
            'cloud computing': 10,
            'machine learning': 9,
            'devops': 9,
            'cybersecurity': 8,
            'mobile development': 8,
            'web development': 7,
            'data science': 9,
            'blockchain': 6,
            'iot': 5
        }
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize skill names for better matching"""
        skill = skill.lower().strip()
        
        # Common normalizations
        normalizations = {
            'js': 'javascript',
            'ts': 'typescript',
            'reactjs': 'react',
            'nodejs': 'node.js',
            'postgres': 'postgresql',
            'k8s': 'kubernetes',
            'aws ec2': 'ec2',
            'amazon web services': 'aws',
            'microsoft azure': 'azure',
            'google cloud platform': 'gcp'
        }
        
        return normalizations.get(skill, skill)
    
    def categorize_skill(self, skill: str) -> str:
        """Determine which category a skill belongs to"""
        skill_normalized = self.normalize_skill(skill)
        
        # Check against known skill categories
        for category, info in self.skill_categories.items():
            if category == 'programming_languages':
                if skill_normalized in info.get('related_skills', {}):
                    return 'programming_languages'
            elif category == 'frameworks_libraries':
                if skill_normalized in info.get('dependencies', {}):
                    return 'frameworks_libraries'
            # Add more specific checks as needed
        
        # Default categorization based on common patterns
        if any(lang in skill_normalized for lang in ['python', 'java', 'javascript', 'c#', 'php', 'ruby', 'go']):
            return 'programming_languages'
        elif any(fw in skill_normalized for fw in ['react', 'angular', 'django', 'spring', 'laravel']):
            return 'frameworks_libraries'
        elif any(db in skill_normalized for db in ['mysql', 'postgresql', 'mongodb', 'redis']):
            return 'databases'
        elif any(cloud in skill_normalized for cloud in ['aws', 'azure', 'gcp', 'cloud']):
            return 'cloud_platforms'
        elif any(devops in skill_normalized for devops in ['docker', 'kubernetes', 'jenkins', 'ci/cd']):
            return 'devops'
        else:
            return 'other'
    
    def calculate_skill_priority(self, skill: str, job_frequency: int = 1) -> int:
        """Calculate priority score for learning a skill (1-10 scale)"""
        skill_normalized = self.normalize_skill(skill)
        category = self.categorize_skill(skill)
        
        # Base priority from category
        category_info = self.skill_categories.get(category, {})
        base_priority = {'high': 8, 'medium': 5, 'low': 3}.get(category_info.get('priority'), 5)
        
        # Adjust based on industry demand
        industry_bonus = 0
        for industry, weight in self.industry_demand.items():
            if any(keyword in skill_normalized for keyword in industry.split()):
                industry_bonus = max(industry_bonus, weight // 2)
        
        # Frequency bonus (how often skill appears in job postings)
        frequency_bonus = min(job_frequency // 10, 3)
        
        # Special high-priority skills
        high_priority_skills = [
            'python', 'javascript', 'react', 'aws', 'docker', 'kubernetes',
            'machine learning', 'data science', 'devops', 'cloud computing'
        ]
        
        if skill_normalized in high_priority_skills:
            base_priority += 2
        
        total_priority = base_priority + industry_bonus + frequency_bonus
        return min(max(total_priority, 1), 10)
    
    def find_related_skills(self, skill: str) -> List[str]:
        """Find skills related to the given skill"""
        skill_normalized = self.normalize_skill(skill)
        related = []
        
        # Check programming language relationships
        prog_langs = self.skill_categories.get('programming_languages', {}).get('related_skills', {})
        if skill_normalized in prog_langs:
            related.extend(prog_langs[skill_normalized])
        
        # Check framework dependencies
        dependencies = self.skill_categories.get('frameworks_libraries', {}).get('dependencies', {})
        for framework, dependency in dependencies.items():
            if skill_normalized == dependency:
                related.append(framework)
            elif skill_normalized == framework:
                related.append(dependency)
        
        # Check cloud platform services
        cloud_services = self.skill_categories.get('cloud_platforms', {}).get('services', {})
        for platform, services in cloud_services.items():
            if skill_normalized == platform:
                related.extend(services[:3])  # Top 3 services
            elif skill_normalized in services:
                related.append(platform)
        
        return list(set(related))
    
    def _generate_ml_recommendations(self, missing_skills: List[str]) -> List[str]:
        """Generate ML-enhanced recommendations for missing skills"""
        recommendations = []
        for skill in missing_skills[:5]:  # Limit to top 5
            recommendations.append(f"Learn {skill} - High priority based on ML analysis")
        return recommendations
    
    def analyze_skill_gaps(self, user_skills: List[str], required_skills: List[str], use_ml: bool = True) -> Dict[str, Any]:
        """Analyze gaps between user skills and required skills using ML-enhanced similarity"""
        if not required_skills:
            return {
                'missing_skills': [],
                'matching_skills': user_skills,
                'skill_coverage': 100.0,
                'recommendations': []
            }
        
        # Use ML-enhanced similarity if available
        if use_ml:
            try:
                from ml_enhanced_analyzer import MLEnhancedAnalyzer
                ml_analyzer = MLEnhancedAnalyzer()
                results = ml_analyzer.enhanced_skill_matching(user_skills, required_skills)
                
                # Convert ML results to expected format
                similarity_score = results.get('tfidf_similarity', 0.0)
                semantic_score = results.get('semantic_similarity', 0.0)
                
                # Use higher threshold for ML-based matching
                ml_threshold = 0.6
                if similarity_score > ml_threshold or semantic_score > ml_threshold:
                    matching_skills = user_skills
                    missing_skills = [skill for skill in required_skills if skill.lower() not in [s.lower() for s in user_skills]]
                else:
                    matching_skills = []
                    missing_skills = required_skills
                
                skill_coverage = (len(matching_skills) / len(required_skills)) * 100 if required_skills else 100.0
                
                return {
                    'missing_skills': missing_skills,
                    'matching_skills': matching_skills,
                    'skill_coverage': skill_coverage,
                    'ml_similarity_score': similarity_score,
                    'ml_semantic_score': semantic_score,
                    'recommendations': self._generate_ml_recommendations(missing_skills)
                }
            except Exception as e:
                logger.warning(f"ML skill gap analysis failed, falling back to rule-based: {e}")
        
        # Fallback to original rule-based approach
        user_skills_normalized = [self.normalize_skill(skill) for skill in user_skills]
        required_skills_normalized = [self.normalize_skill(skill) for skill in required_skills]
        
        # Find matching and missing skills
        matching_skills = []
        missing_skills = []
        
        for req_skill in required_skills_normalized:
            found_match = False
            for user_skill in user_skills_normalized:
                if self._skills_match(user_skill, req_skill):
                    matching_skills.append(req_skill)
                    found_match = True
                    break
            
            if not found_match:
                missing_skills.append(req_skill)
        
        # Find partial matches (related skills)
        partial_matches = []
        for required_skill in missing_skills[:]:  # Copy list to modify during iteration
            for user_skill in user_skills_normalized:
                related_skills = self.find_related_skills(user_skill)
                if required_skill in [self.normalize_skill(rs) for rs in related_skills]:
                    partial_matches.append({
                        'required_skill': required_skill,
                        'user_has': user_skill,
                        'relationship': 'related'
                    })
                    # Remove from missing since it's a partial match
                    if required_skill in missing_skills:
                        missing_skills.remove(required_skill)
        
        # Prioritize missing skills
        prioritized_missing = []
        for skill in missing_skills:
            priority = self.calculate_skill_priority(skill)
            category = self.categorize_skill(skill)
            related = self.find_related_skills(skill)
            
            prioritized_missing.append({
                'skill': skill,
                'priority': priority,
                'category': category,
                'related_skills': related[:5],  # Limit to 5 related skills
                'learning_difficulty': self._estimate_learning_difficulty(skill, user_skills_normalized)
            })
        
        # Sort by priority (descending)
        prioritized_missing.sort(key=lambda x: x['priority'], reverse=True)
        
        # Generate learning path recommendations
        learning_path = self._generate_learning_path(prioritized_missing[:8])  # Top 8 skills
        
        # Calculate match percentage
        total_required = len(required_skills_normalized)
        if total_required > 0:
            match_percentage = ((len(matching_skills) + len(partial_matches) * 0.5) / total_required) * 100
        else:
            match_percentage = 100.0
        
        analysis = {
            'match_percentage': round(match_percentage, 2),
            'matching_skills': matching_skills,
            'missing_skills': [skill['skill'] for skill in prioritized_missing],
            'missing_skills_detailed': prioritized_missing,
            'partial_matches': partial_matches,
            'learning_path': learning_path,
            'recommendations': self._generate_recommendations(prioritized_missing, user_skills_normalized),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Save analysis for trends
        self._save_skill_analysis(analysis)
        
        return analysis
    
    def _estimate_learning_difficulty(self, skill: str, user_skills: List[str]) -> str:
        """Estimate learning difficulty based on user's existing skills"""
        skill_normalized = self.normalize_skill(skill)
        category = self.categorize_skill(skill)
        
        # Check if user has related skills
        related_skills = self.find_related_skills(skill_normalized)
        has_related = any(related in user_skills for related in related_skills)
        
        # Base difficulty by category
        category_difficulty = {
            'programming_languages': 'medium',
            'frameworks_libraries': 'easy' if has_related else 'medium',
            'databases': 'easy',
            'cloud_platforms': 'medium',
            'devops': 'hard',
            'soft_skills': 'easy'
        }
        
        base_difficulty = category_difficulty.get(category, 'medium')
        
        # Adjust based on prerequisites
        if skill_normalized in ['kubernetes', 'terraform', 'machine learning']:
            return 'hard'
        elif skill_normalized in ['docker', 'git', 'html', 'css']:
            return 'easy'
        
        return base_difficulty
    
    def _generate_learning_path(self, missing_skills: List[Dict]) -> List[Dict[str, Any]]:
        """Generate optimized learning path for missing skills"""
        if not missing_skills:
            return []
        
        # Group by category and difficulty
        learning_groups = {
            'foundation': [],  # Easy skills that are prerequisites
            'core': [],        # Important medium-difficulty skills
            'advanced': []     # Complex skills requiring strong foundation
        }
        
        for skill_info in missing_skills:
            difficulty = skill_info['learning_difficulty']
            priority = skill_info['priority']
            
            if difficulty == 'easy' or priority >= 8:
                learning_groups['foundation'].append(skill_info)
            elif difficulty == 'medium' or priority >= 6:
                learning_groups['core'].append(skill_info)
            else:
                learning_groups['advanced'].append(skill_info)
        
        # Create ordered learning path
        learning_path = []
        phase = 1
        
        for group_name, skills in learning_groups.items():
            if not skills:
                continue
                
            # Sort skills within group by priority
            skills.sort(key=lambda x: x['priority'], reverse=True)
            
            learning_path.append({
                'phase': phase,
                'phase_name': group_name.title(),
                'description': self._get_phase_description(group_name),
                'skills': skills[:4],  # Limit to 4 skills per phase
                'estimated_duration': self._estimate_phase_duration(skills[:4]),
                'prerequisites': self._get_phase_prerequisites(group_name)
            })
            phase += 1
        
        return learning_path
    
    def _get_phase_description(self, phase_name: str) -> str:
        """Get description for learning phase"""
        descriptions = {
            'foundation': "Essential skills that form the foundation for advanced learning",
            'core': "Important skills that are in high demand and build upon foundation skills",
            'advanced': "Specialized skills that require solid understanding of core concepts"
        }
        return descriptions.get(phase_name, "Skills development phase")
    
    def _estimate_phase_duration(self, skills: List[Dict]) -> str:
        """Estimate time needed to complete a learning phase"""
        if not skills:
            return "0 weeks"
        
        total_weeks = 0
        for skill in skills:
            difficulty = skill['learning_difficulty']
            if difficulty == 'easy':
                total_weeks += 2
            elif difficulty == 'medium':
                total_weeks += 4
            else:  # hard
                total_weeks += 6
        
        # Account for parallel learning
        avg_weeks = total_weeks // len(skills) if len(skills) > 1 else total_weeks
        return f"{avg_weeks}-{total_weeks} weeks"
    
    def _get_phase_prerequisites(self, phase_name: str) -> List[str]:
        """Get prerequisites for a learning phase"""
        prerequisites = {
            'foundation': ["Basic computer skills", "Problem-solving mindset"],
            'core': ["Completion of foundation skills", "Hands-on practice experience"],
            'advanced': ["Strong foundation in core skills", "Project experience", "Understanding of software architecture"]
        }
        return prerequisites.get(phase_name, [])
    
    def _generate_recommendations(self, missing_skills: List[Dict], user_skills: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        if not missing_skills:
            return ["Great! You have all the required skills. Focus on gaining practical experience."]
        
        recommendations = []
        
        # High priority skills
        high_priority = [s for s in missing_skills if s['priority'] >= 8]
        if high_priority:
            top_skill = high_priority[0]['skill']
            recommendations.append(f"Start with {top_skill} - it's high priority and in demand")
        
        # Foundation skills
        easy_skills = [s for s in missing_skills if s['learning_difficulty'] == 'easy']
        if easy_skills and len(easy_skills) >= 2:
            recommendations.append("Focus on building foundational skills first before moving to advanced topics")
        
        # Related skills to user's existing skills
        for user_skill in user_skills[:5]:  # Check top 5 user skills
            related = self.find_related_skills(user_skill)
            missing_related = [s for s in missing_skills if s['skill'] in related]
            if missing_related:
                skill_name = missing_related[0]['skill']
                recommendations.append(f"Learn {skill_name} - it's closely related to your {user_skill} skills")
                break
        
        # Category-specific advice
        categories = {}
        for skill in missing_skills[:5]:
            cat = skill['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(skill['skill'])
        
        for category, skills in categories.items():
            if category == 'cloud_platforms' and len(skills) >= 2:
                recommendations.append("Consider getting cloud platform certification to validate your skills")
            elif category == 'devops' and len(skills) >= 2:
                recommendations.append("Set up personal projects to practice DevOps tools and workflows")
        
        # General advice
        if len(missing_skills) > 6:
            recommendations.append("Focus on 2-3 skills at a time rather than trying to learn everything at once")
        
        recommendations.append("Combine learning with hands-on projects for better retention and portfolio building")
        
        return recommendations[:6]  # Limit to 6 recommendations
    
    def _save_skill_analysis(self, analysis: Dict[str, Any]) -> None:
        """Save skill analysis for trend tracking"""
        try:
            trend_data = {
                'timestamp': analysis['analysis_timestamp'],
                'match_percentage': analysis['match_percentage'],
                'missing_skills_count': len(analysis['missing_skills']),
                'top_missing_skills': analysis['missing_skills'][:5],
                'categories_needed': {}
            }
            
            # Count missing skills by category
            for skill_detail in analysis['missing_skills_detailed']:
                category = skill_detail['category']
                trend_data['categories_needed'][category] = trend_data['categories_needed'].get(category, 0) + 1
            
            # Load existing trends
            if os.path.exists(self.skill_trends_file):
                with open(self.skill_trends_file, 'r', encoding='utf-8') as f:
                    trends = json.load(f)
            else:
                trends = {'analyses': []}
            
            trends['analyses'].append(trend_data)
            
            # Keep only last 100 analyses
            trends['analyses'] = trends['analyses'][-100:]
            
            # Save updated trends
            with open(self.skill_trends_file, 'w', encoding='utf-8') as f:
                json.dump(trends, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving skill analysis trends: {e}")
    
    def get_skill_trends(self) -> Dict[str, Any]:
        """Get skill trends from historical analyses"""
        try:
            if not os.path.exists(self.skill_trends_file):
                return {'message': 'No trend data available yet'}
            
            with open(self.skill_trends_file, 'r', encoding='utf-8') as f:
                trends = json.load(f)
            
            analyses = trends.get('analyses', [])
            if not analyses:
                return {'message': 'No trend data available yet'}
            
            # Calculate trends
            total_analyses = len(analyses)
            avg_match_percentage = sum(a['match_percentage'] for a in analyses) / total_analyses
            
            # Most commonly missing skills
            all_missing_skills = []
            for analysis in analyses:
                all_missing_skills.extend(analysis['top_missing_skills'])
            
            missing_skill_counts = Counter(all_missing_skills)
            
            # Category trends
            category_trends = {}
            for analysis in analyses:
                for category, count in analysis['categories_needed'].items():
                    if category not in category_trends:
                        category_trends[category] = []
                    category_trends[category].append(count)
            
            # Calculate category averages
            category_averages = {}
            for category, counts in category_trends.items():
                category_averages[category] = sum(counts) / len(counts)
            
            return {
                'total_analyses': total_analyses,
                'average_skill_match': round(avg_match_percentage, 2),
                'most_missing_skills': dict(missing_skill_counts.most_common(10)),
                'category_gap_trends': category_averages,
                'recent_trend': 'improving' if analyses[-1]['match_percentage'] > avg_match_percentage else 'needs_attention'
            }
            
        except Exception as e:
            return {'error': f'Error retrieving skill trends: {str(e)}'}
    
    def compare_skill_profiles(self, profile1_skills: List[str], profile2_skills: List[str]) -> Dict[str, Any]:
        """Compare two skill profiles"""
        # Normalize skills
        skills1 = set(self.normalize_skill(skill) for skill in profile1_skills)
        skills2 = set(self.normalize_skill(skill) for skill in profile2_skills)
        
        # Find overlaps and differences
        common_skills = skills1 & skills2
        profile1_unique = skills1 - skills2
        profile2_unique = skills2 - skills1
        
        # Calculate similarity score
        total_unique_skills = len(skills1 | skills2)
        similarity_score = len(common_skills) / total_unique_skills * 100 if total_unique_skills > 0 else 0
        
        # Categorize skills
        def categorize_skills(skills_set):
            categories = {}
            for skill in skills_set:
                category = self.categorize_skill(skill)
                if category not in categories:
                    categories[category] = []
                categories[category].append(skill)
            return categories
        
        return {
            'similarity_score': round(similarity_score, 2),
            'common_skills': list(common_skills),
            'profile1_unique': list(profile1_unique),
            'profile2_unique': list(profile2_unique),
            'common_skills_by_category': categorize_skills(common_skills),
            'profile1_strengths': categorize_skills(profile1_unique),
            'profile2_strengths': categorize_skills(profile2_unique),
            'recommendation': self._get_profile_comparison_recommendation(similarity_score)
        }
    
    def _get_profile_comparison_recommendation(self, similarity_score: float) -> str:
        """Get recommendation based on profile similarity"""
        if similarity_score >= 80:
            return "Very similar skill sets. Consider specializing in different areas to complement each other."
        elif similarity_score >= 60:
            return "Good skill overlap with some unique strengths. Great for collaborative projects."
        elif similarity_score >= 40:
            return "Moderate overlap. Each profile has distinct strengths that could be complementary."
        else:
            return "Different skill sets. Could benefit from knowledge sharing and cross-training."
    
    def get_learning_resources(self, skill: str) -> Dict[str, Any]:
        """Get recommended learning resources for a skill"""
        skill_normalized = self.normalize_skill(skill)
        category = self.categorize_skill(skill_normalized)
        difficulty = self._estimate_learning_difficulty(skill_normalized, [])
        
        # General resource types based on skill category
        resource_suggestions = {
            'programming_languages': {
                'online_courses': ['Codecademy', 'freeCodeCamp', 'Coursera', 'edX'],
                'practice_platforms': ['LeetCode', 'HackerRank', 'Codewars'],
                'documentation': ['Official documentation', 'MDN Web Docs'],
                'projects': ['Build a personal website', 'Create a small application', 'Contribute to open source']
            },
            'frameworks_libraries': {
                'online_courses': ['Udemy', 'Pluralsight', 'YouTube tutorials'],
                'practice_platforms': ['CodeSandbox', 'GitHub', 'Repl.it'],
                'documentation': ['Official framework documentation', 'Community tutorials'],
                'projects': ['Build a portfolio project', 'Clone popular applications', 'Follow framework tutorials']
            },
            'cloud_platforms': {
                'online_courses': ['AWS Training', 'Azure Learn', 'Google Cloud Skills Boost'],
                'practice_platforms': ['AWS Free Tier', 'Azure Free Account', 'GCP Free Tier'],
                'certifications': ['AWS Solutions Architect', 'Azure Fundamentals', 'GCP Associate'],
                'projects': ['Deploy a web application', 'Set up CI/CD pipeline', 'Create cloud infrastructure']
            },
            'devops': {
                'online_courses': ['Docker courses', 'Kubernetes tutorials', 'DevOps bootcamps'],
                'practice_platforms': ['Play with Docker', 'Katacoda', 'KillerCoda'],
                'tools': ['Set up local environment', 'Practice with containers', 'Learn infrastructure as code'],
                'projects': ['Containerize an application', 'Set up monitoring', 'Automate deployments']
            }
        }
        
        resources = resource_suggestions.get(category, {
            'online_courses': ['Search for courses on major platforms'],
            'practice_platforms': ['Practice with hands-on exercises'],
            'projects': ['Apply skill in real-world projects']
        })
        
        # Add estimated learning time
        time_estimates = {
            'easy': '2-4 weeks with consistent practice',
            'medium': '1-3 months with regular learning',
            'hard': '3-6 months with dedicated study'
        }
        
        return {
            'skill': skill,
            'category': category,
            'difficulty': difficulty,
            'estimated_learning_time': time_estimates.get(difficulty, '1-3 months'),
            'recommended_resources': resources,
            'prerequisites': self.find_related_skills(skill_normalized),
            'next_skills': self.find_related_skills(skill_normalized)
        }