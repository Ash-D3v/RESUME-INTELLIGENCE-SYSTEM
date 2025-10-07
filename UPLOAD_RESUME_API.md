# POST /upload-resume - API Documentation

## Overview

The `/upload-resume` endpoint is the main entry point for comprehensive resume analysis. It accepts a resume file (PDF, DOCX, or TXT) and returns detailed analysis including skill extraction, ATS scoring, job matching, skill gap analysis, and course recommendations.

---

## Endpoint Details

**URL:** `POST /upload-resume`  
**Content-Type:** `multipart/form-data`  
**Authentication:** None (currently)

---

## Input Parameters

### Required Parameters

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `file` | File | Form Data | Resume file (PDF, DOCX, or TXT) |

### Optional Parameters

| Parameter | Type | Location | Description | Default |
|-----------|------|----------|-------------|---------|
| `job_description` | string | Query String | Job description for targeted analysis | "" (empty) |

### Supported File Types

- **PDF:** `application/pdf`
- **DOCX:** `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
- **TXT:** `text/plain`

---

## Request Examples

### Example 1: Basic Upload (No Job Description)

**cURL:**
```bash
curl -X POST "http://localhost:8000/upload-resume" \
  -F "file=@/path/to/resume.pdf"
```

**JavaScript (Fetch API):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/upload-resume', {
  method: 'POST',
  body: formData
});

const result = await response.json();
```

**Python (requests):**
```python
import requests

files = {'file': open('resume.pdf', 'rb')}
response = requests.post('http://localhost:8000/upload-resume', files=files)
result = response.json()
```

### Example 2: Upload with Job Description

**cURL:**
```bash
curl -X POST "http://localhost:8000/upload-resume?job_description=Looking%20for%20Python%20developer%20with%203%20years%20experience" \
  -F "file=@/path/to/resume.pdf"
```

**JavaScript (Fetch API):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const jobDescription = "Looking for Python developer with 3 years experience";
const url = `http://localhost:8000/upload-resume?job_description=${encodeURIComponent(jobDescription)}`;

const response = await fetch(url, {
  method: 'POST',
  body: formData
});

const result = await response.json();
```

**React Example:**
```typescript
import apiService from "@/lib/api";

const handleFileUpload = async (file: File, jobDescription?: string) => {
  try {
    const result = await apiService.uploadResume(file, jobDescription);
    console.log("Analysis complete:", result);
    
    // Access different parts of the response
    console.log("ATS Score:", result.ats_analysis.scores.overallScore);
    console.log("Skills Found:", result.resume_analysis.total_skills_found);
    console.log("Job Matches:", result.job_recommendations.matching_jobs);
  } catch (error) {
    console.error("Upload failed:", error);
  }
};
```

---

## Response Structure

### Success Response (200 OK)

```json
{
  "success": true,
  "timestamp": "2025-10-06T11:08:19.123456",
  "filename": "john_doe_resume.pdf",
  "file_size": 245678,
  
  "resume_analysis": {
    "extracted_text": "Full resume text here...",
    "parsed_resume": {
      "name": "John Doe",
      "email": "john.doe@example.com",
      "phone": "+1-234-567-8900",
      "experience": "5 years",
      "education": "Bachelor of Science in Computer Science"
    },
    "extracted_skills": {
      "programming_languages": ["Python", "JavaScript", "Java"],
      "frameworks": ["React", "Django", "Node.js"],
      "databases": ["PostgreSQL", "MongoDB"],
      "tools": ["Git", "Docker", "AWS"]
    },
    "total_skills_found": 15,
    "skills_by_category": {
      "programming_languages": 3,
      "frameworks": 3,
      "databases": 2,
      "tools": 3
    }
  },
  
  "ats_analysis": {
    "scores": {
      "overallScore": 85,
      "structureScore": 90,
      "keywordScore": 80,
      "formatScore": 85
    },
    "score_breakdown": {
      "strengths": [
        "Well-structured resume",
        "Good keyword density",
        "Clear formatting"
      ],
      "improvements": [
        "Add more quantifiable achievements",
        "Include relevant certifications"
      ]
    },
    "job_description_provided": true
  },
  
  "skill_gap_analysis": {
    "missing_skills": ["Kubernetes", "TypeScript", "GraphQL"],
    "matching_skills": ["Python", "JavaScript", "React", "Docker"],
    "gap_percentage": 20,
    "priority_skills": [
      {
        "skill": "Kubernetes",
        "priority": "High",
        "reason": "Required for DevOps roles"
      }
    ]
  },
  
  "job_recommendations": {
    "matching_jobs": [
      {
        "title": "Senior Python Developer",
        "company": "Tech Innovations Inc",
        "location": "Mumbai, Maharashtra",
        "experience": "3-5 years",
        "salary": "15-25 LPA",
        "relevance": 85.5,
        "requiredSkills": ["python", "django", "postgresql", "docker"],
        "description": "We are looking for an experienced Python developer...",
        "qualification": "BTech/BE in Computer Science",
        "job_id": 1001,
        "matching_skills": ["python", "django", "docker"]
      }
    ],
    "total_matches": 5,
    "search_based_on_skills": ["python", "javascript", "react", "django"]
  },
  
  "course_recommendations": {
    "targeted_courses": {
      "kubernetes": [
        {
          "title": "Kubernetes Complete Course",
          "provider": "Udemy",
          "url": "https://www.udemy.com/courses/search/?q=kubernetes",
          "description": "Master kubernetes from scratch...",
          "duration": "20-30 hours",
          "rating": "4.5",
          "price": "Paid (Often on Sale)",
          "level": "All Levels",
          "skills_covered": ["kubernetes"]
        }
      ]
    },
    "general_courses": {
      "python": [
        {
          "title": "Python Complete Course",
          "provider": "ML-Enhanced Platform",
          "url": "https://ml-platform.com/course/python",
          "description": "ML-recommended course for python...",
          "duration": "Self-paced",
          "rating": "4.5",
          "price": "Subscription",
          "level": "Adaptive",
          "skills_covered": ["python"]
        }
      ]
    },
    "popular_courses": [
      {
        "title": "Machine Learning by Andrew Ng",
        "provider": "Coursera",
        "rating": 4.9,
        "enrollment": "4.5M+",
        "category": "machine learning"
      }
    ],
    "total_courses": 12
  },
  
  "insights": {
    "skill_trends": {
      "trending_up": ["Kubernetes", "TypeScript", "GraphQL"],
      "stable": ["Python", "JavaScript", "React"],
      "declining": []
    },
    "recommendations": [
      "Great ATS score! Your resume is well-optimized",
      "Consider learning Kubernetes, TypeScript, GraphQL to match job requirements",
      "Found 5 relevant job opportunities based on your skills"
    ]
  }
}
```

### Error Responses

#### 400 Bad Request - Invalid File Type
```json
{
  "detail": "Only PDF, DOCX, and TXT files are supported"
}
```

#### 400 Bad Request - Empty File
```json
{
  "detail": "No text could be extracted from the file"
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Error processing resume: [error message]"
}
```

---

## Response Fields Explained

### Top Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Indicates if the request was successful |
| `timestamp` | string | ISO 8601 timestamp of when the analysis was performed |
| `filename` | string | Original filename of the uploaded resume |
| `file_size` | integer | Size of the uploaded file in bytes |

### resume_analysis Object

| Field | Type | Description |
|-------|------|-------------|
| `extracted_text` | string | Full text extracted from the resume |
| `parsed_resume` | object | Structured data extracted (name, email, phone, etc.) |
| `extracted_skills` | object | Skills categorized by type |
| `total_skills_found` | integer | Total number of unique skills identified |
| `skills_by_category` | object | Count of skills per category |

### ats_analysis Object

| Field | Type | Description |
|-------|------|-------------|
| `scores` | object | ATS scoring metrics (0-100) |
| `scores.overallScore` | integer | Overall ATS compatibility score |
| `scores.structureScore` | integer | Resume structure quality score |
| `scores.keywordScore` | integer | Keyword matching score |
| `scores.formatScore` | integer | Formatting quality score |
| `score_breakdown` | object | Detailed analysis with strengths and improvements |
| `job_description_provided` | boolean | Whether job description was included in request |

### skill_gap_analysis Object

| Field | Type | Description |
|-------|------|-------------|
| `missing_skills` | array | Skills required but not present in resume |
| `matching_skills` | array | Skills that match job requirements |
| `gap_percentage` | number | Percentage of required skills missing |
| `priority_skills` | array | Missing skills ranked by importance |

### job_recommendations Object

| Field | Type | Description |
|-------|------|-------------|
| `matching_jobs` | array | List of relevant job opportunities |
| `total_matches` | integer | Total number of matching jobs found |
| `search_based_on_skills` | array | Skills used for job matching |

### course_recommendations Object

| Field | Type | Description |
|-------|------|-------------|
| `targeted_courses` | object | Courses for missing skills (skill gap) |
| `general_courses` | object | Courses for existing skills (upskilling) |
| `popular_courses` | array | Popular courses across all categories |
| `total_courses` | integer | Total number of courses recommended |

### insights Object

| Field | Type | Description |
|-------|------|-------------|
| `skill_trends` | object | Current market trends for skills |
| `recommendations` | array | Actionable recommendations for improvement |

---

## Frontend Integration

### React Component Example

```typescript
import { useState } from "react";
import apiService from "@/lib/api";
import { toast } from "sonner";

export default function ResumeUpload() {
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileUpload = async (file: File) => {
    setAnalyzing(true);
    toast.loading("Analyzing your resume...", { id: "analysis" });

    try {
      const analysisResult = await apiService.uploadResume(file);
      setResult(analysisResult);
      toast.success("Analysis complete!", { id: "analysis" });
      
      // Display results
      console.log("ATS Score:", analysisResult.ats_analysis.scores.overallScore);
      console.log("Skills:", analysisResult.resume_analysis.total_skills_found);
      console.log("Jobs:", analysisResult.job_recommendations.total_matches);
      
    } catch (error: any) {
      toast.error(error.message || "Failed to analyze resume", { id: "analysis" });
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div>
      <input
        type="file"
        accept=".pdf,.docx,.txt"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFileUpload(file);
        }}
        disabled={analyzing}
      />
      
      {result && (
        <div>
          <h2>ATS Score: {result.ats_analysis.scores.overallScore}/100</h2>
          <p>Skills Found: {result.resume_analysis.total_skills_found}</p>
          <p>Job Matches: {result.job_recommendations.total_matches}</p>
        </div>
      )}
    </div>
  );
}
```

---

## Testing

### Test with cURL

```bash
# Test with a sample resume
curl -X POST "http://localhost:8000/upload-resume" \
  -F "file=@sample_resume.pdf" \
  | jq '.'

# Test with job description
curl -X POST "http://localhost:8000/upload-resume?job_description=Python%20Developer%20with%20Django" \
  -F "file=@sample_resume.pdf" \
  | jq '.ats_analysis.scores'
```

### Test with Postman

1. Set method to **POST**
2. URL: `http://localhost:8000/upload-resume`
3. Go to **Body** tab
4. Select **form-data**
5. Add key `file` with type **File**
6. Choose your resume file
7. (Optional) Add query parameter `job_description`
8. Click **Send**

---

## Performance

- **Average Response Time:** 2-5 seconds (depending on file size and complexity)
- **Max File Size:** No explicit limit (recommended < 10MB)
- **Concurrent Requests:** Supported (FastAPI async)
- **Timeout:** 30 seconds (configurable)

---

## Notes

1. **File Processing:** PDF and DOCX files are parsed to extract text. Ensure files are not password-protected or corrupted.

2. **Skill Extraction:** Uses NLP and pattern matching to identify skills. May not catch all skills if they're in unusual formats.

3. **Job Matching:** Currently uses fallback mock data (10 jobs) due to external API unavailability. Will return 5 most relevant matches.

4. **Course Recommendations:** ML-enhanced recommendations from multiple platforms (Udemy, Coursera, edX, Pluralsight).

5. **Caching:** Resume data is saved locally for analytics purposes.

---

## Error Handling

The endpoint includes comprehensive error handling:

- **File validation** (type, size, content)
- **Text extraction** (PDF/DOCX parsing)
- **Graceful fallbacks** (if external services fail)
- **Detailed error messages** (for debugging)

---

## Related Endpoints

- **POST /score** - Score resume against job description only
- **GET /jobs** - Get job matches by skills
- **GET /skill-gap** - Analyze skill gaps
- **GET /courses** - Get course recommendations
- **POST /detailed-analysis** - Alternative analysis endpoint

---

## Changelog

**v1.0.0 (Current)**
- Initial release
- Comprehensive resume analysis
- ATS scoring
- Job matching with fallback data
- ML-enhanced course recommendations
- Skill gap analysis

---

## Support

For issues or questions:
- Check API documentation: http://localhost:8000/docs
- View logs: `docker-compose logs -f backend`
- Test connection: http://localhost:8080/connection-test
