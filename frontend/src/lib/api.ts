// API service for communicating with the backend
const API_BASE_URL = 'http://localhost:8000';

export interface ResumeAnalysisResponse {
  success: boolean;
  timestamp: string;
  filename: string;
  file_size: number;
  resume_analysis: {
    extracted_text: string;
    parsed_resume: {
      contact_info: {
        email: string;
        phone: string;
        linkedin: string;
      };
      education: Array<{
        degree: string;
        context: string;
      }>;
      experience: any[];
      skills: {
        programming_languages: string[];
        frameworks_libraries: string[];
        tools_technologies: string[];
        cloud_platforms: string[];
        methodologies: string[];
        soft_skills: string[];
        web_technologies: string[];
        data_science: string[];
        devops: string[];
      };
      certifications: any[];
      summary: string;
      projects: any[];
    };
    extracted_skills: {
      programming_languages: string[];
      frameworks_libraries: string[];
      tools_technologies: string[];
      cloud_platforms: string[];
      methodologies: string[];
      soft_skills: string[];
      web_technologies: string[];
      data_science: string[];
      devops: string[];
    };
    total_skills_found: number;
    skills_by_category: Record<string, number>;
  };
  ats_analysis: {
    scores: {
      overallScore: number;
      structureScore: number;
      keywordScore: number;
      formatScore: number;
      jobMatchScore: number;
    };
    score_breakdown: {
      scores: {
        overallScore: number;
        structureScore: number;
        keywordScore: number;
        formatScore: number;
        jobMatchScore: number;
      };
      feedback: {
        overall_feedback: string;
        structure_feedback: string;
        keyword_feedback: string;
        format_feedback: string;
        job_match_feedback: string;
      };
      recommendations: string[];
    };
    job_description_provided: boolean;
  };
  skill_gap_analysis: {
    missing_skills: string[];
    matching_skills: string[];
    skill_coverage: number;
    recommendations: string[];
  };
  job_recommendations: {
    matching_jobs: Array<{
      title: string;
      company: string;
      location: string;
      experience: string;
      salary: string;
      relevance: number;
      requiredSkills: string[];
      description: string;
      qualification: string;
      job_id: number;
      matching_skills: string[];
    }>;
    total_matches: number;
    search_based_on_skills: string[];
  };
  course_recommendations: {
    targeted_courses: Record<string, any[]>;
    general_courses: Record<string, any[]>;
    popular_courses: Array<{
      title: string;
      provider: string;
      rating: number;
      enrollment: string;
      category: string;
    }>;
    total_courses: number;
  };
  insights: {
    skill_trends: {
      total_analyses: number;
      average_skill_match: number;
      most_missing_skills: Record<string, number>;
      category_gap_trends: Record<string, number>;
      recent_trend: string;
    };
    recommendations: string[];
  };
}

export class ApiService {
  private static baseUrl = API_BASE_URL;

  static async uploadResume(file: File, jobDescription?: string): Promise<ResumeAnalysisResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    if (jobDescription) {
      formData.append('job_description', jobDescription);
    }

    try {
      const response = await fetch(`${this.baseUrl}/upload-resume`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error uploading resume:', error);
      throw error;
    }
  }

  static async scoreResume(resumeText: string, jobDescription?: string): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/score`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          resume_text: resumeText,
          job_description: jobDescription || '',
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error scoring resume:', error);
      throw error;
    }
  }

  static async getJobs(skills: string[]): Promise<any> {
    try {
      const skillsParam = skills.join(',');
      const response = await fetch(`${this.baseUrl}/jobs?skills=${encodeURIComponent(skillsParam)}`);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching jobs:', error);
      throw error;
    }
  }

  static async getCourses(skills: string[]): Promise<any> {
    try {
      const skillsParam = skills.join(',');
      const response = await fetch(`${this.baseUrl}/courses?skills=${encodeURIComponent(skillsParam)}`);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching courses:', error);
      throw error;
    }
  }

  static async getAnalytics(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/analytics`);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching analytics:', error);
      throw error;
    }
  }

  static async healthCheck(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Backend health check failed:', error);
      throw error;
    }
  }
}
