import { useState } from "react";
import FileUpload from "@/components/FileUpload";
import ProgressCircle from "@/components/ProgressCircle";
import StatCard from "@/components/StatCard";
import SkillBadge from "@/components/SkillBadge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FileText, Briefcase, Target, TrendingUp, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { ApiService, ResumeAnalysisResponse } from "@/lib/api";

export default function Dashboard() {
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [analysisData, setAnalysisData] = useState<ResumeAnalysisResponse | null>(null);

  const handleFileSelect = async (file: File) => {
    setAnalyzing(true);
    toast.loading("Analyzing your resume...", { id: "analysis" });

    try {
      const response = await ApiService.uploadResume(file);
      setAnalysisData(response);
      setAnalysisComplete(true);
      toast.success("Analysis complete!", { id: "analysis" });
    } catch (error) {
      console.error('Analysis failed:', error);
      toast.error(error instanceof Error ? error.message : "Failed to analyze resume", { id: "analysis" });
    } finally {
      setAnalyzing(false);
    }
  };

  // Get real data from analysis or fallback to defaults
  const displayData = analysisData ? {
    atsScore: analysisData.ats_analysis.scores.overallScore,
    skillsExtracted: analysisData.resume_analysis.total_skills_found,
    matchedJobs: analysisData.job_recommendations.total_matches,
    skillGaps: analysisData.skill_gap_analysis.missing_skills.length,
    skills: Object.entries(analysisData.resume_analysis.extracted_skills).flatMap(([category, skills]) =>
      skills.map(skill => ({ name: skill, level: "intermediate" as const }))
    ).slice(0, 10), // Limit to first 10 skills
    contact: {
      name: analysisData.resume_analysis.parsed_resume.contact_info.email.split('@')[0] || "User",
      email: analysisData.resume_analysis.parsed_resume.contact_info.email,
      phone: analysisData.resume_analysis.parsed_resume.contact_info.phone,
    },
    experience: analysisData.resume_analysis.parsed_resume.experience.length > 0 ? 
      `${analysisData.resume_analysis.parsed_resume.experience.length} experience entries` : "No experience listed",
    education: analysisData.resume_analysis.parsed_resume.education.length > 0 ? 
      analysisData.resume_analysis.parsed_resume.education[0].degree : "No education listed",
  } : {
    atsScore: 0,
    skillsExtracted: 0,
    matchedJobs: 0,
    skillGaps: 0,
    skills: [],
    contact: { name: "", email: "", phone: "" },
    experience: "",
    education: "",
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Hero Section */}
      <div className="text-center space-y-4">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 animate-scale-in">
          <Sparkles className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-primary">AI-Powered Resume Analysis</span>
        </div>
        <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent">
          Upload Your Resume
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Get instant ATS scoring, skill extraction, job matching, and personalized career guidance
        </p>
      </div>

      {/* Upload Section */}
      <div className="max-w-3xl mx-auto">
        <FileUpload onFileSelect={handleFileSelect} />
        {analyzing && (
          <div className="mt-6 text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20">
              <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
              <span className="text-sm font-medium text-primary">Processing your resume...</span>
            </div>
          </div>
        )}
      </div>

      {/* Analysis Results */}
      {analysisComplete && (
        <div className="space-y-8 animate-slide-up">
          {/* Stats Overview */}
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            <StatCard
              title="ATS Score"
              value={displayData.atsScore}
              icon={Target}
              variant="primary"
              description="Applicant Tracking System compatibility"
            />
            <StatCard
              title="Skills Found"
              value={displayData.skillsExtracted}
              icon={Sparkles}
              variant="success"
              description="Extracted from your resume"
            />
            <StatCard
              title="Matched Jobs"
              value={displayData.matchedJobs}
              icon={Briefcase}
              variant="default"
              description="Based on your skills"
            />
            <StatCard
              title="Skill Gaps"
              value={displayData.skillGaps}
              icon={TrendingUp}
              variant="warning"
              description="Areas for improvement"
            />
          </div>

          {/* Detailed Analysis */}
          <div className="grid gap-6 lg:grid-cols-3">
            {/* ATS Score Card */}
            <Card className="glass-card border shadow-lg">
              <CardHeader>
                <CardTitle>ATS Score Analysis</CardTitle>
                <CardDescription>How well your resume passes ATS systems</CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col items-center gap-4">
                <ProgressCircle value={displayData.atsScore} size={140} />
                <div className="text-center space-y-2">
                  <p className="text-sm text-muted-foreground">
                    {analysisData?.ats_analysis.score_breakdown.feedback.overall_feedback || 
                     (displayData.atsScore >= 85
                      ? "Excellent! Your resume is well-optimized."
                      : displayData.atsScore >= 70
                      ? "Good score. Some improvements recommended."
                      : "Needs improvement for better ATS compatibility.")}
                  </p>
                  <Button className="w-full shadow-md">View Details</Button>
                </div>
              </CardContent>
            </Card>

            {/* Contact Info */}
            <Card className="glass-card border shadow-lg">
              <CardHeader>
                <CardTitle>Contact Information</CardTitle>
                <CardDescription>Extracted from your resume</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <p className="text-xs font-medium text-muted-foreground">Name</p>
                  <p className="text-sm font-semibold">{displayData.contact.name || "Not found"}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-muted-foreground">Email</p>
                  <p className="text-sm font-semibold">{displayData.contact.email || "Not found"}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-muted-foreground">Phone</p>
                  <p className="text-sm font-semibold">{displayData.contact.phone || "Not found"}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-muted-foreground">Experience</p>
                  <p className="text-sm font-semibold">{displayData.experience || "Not found"}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-muted-foreground">Education</p>
                  <p className="text-sm font-semibold">{displayData.education || "Not found"}</p>
                </div>
              </CardContent>
            </Card>

            {/* Skills */}
            <Card className="glass-card border shadow-lg">
              <CardHeader>
                <CardTitle>Extracted Skills</CardTitle>
                <CardDescription>Skills identified from your resume</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {displayData.skills.map((skill) => (
                    <SkillBadge key={skill.name} skill={skill.name} level={skill.level} />
                  ))}
                </div>
                <Button variant="outline" className="w-full mt-4">
                  <FileText className="w-4 h-4 mr-2" />
                  Export Skills List
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Recommendations */}
          <Card className="glass-card border shadow-lg">
            <CardHeader>
              <CardTitle>AI Recommendations</CardTitle>
              <CardDescription>Personalized suggestions to improve your profile</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="jobs" className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="jobs">Matched Jobs</TabsTrigger>
                  <TabsTrigger value="gaps">Skill Gaps</TabsTrigger>
                  <TabsTrigger value="courses">Courses</TabsTrigger>
                </TabsList>
                <TabsContent value="jobs" className="space-y-4 mt-4">
                  <div className="space-y-3">
                    {analysisData?.job_recommendations.matching_jobs.slice(0, 3).map((job) => (
                      <div key={job.job_id} className="p-4 rounded-lg border bg-card/50 hover:bg-card transition-colors">
                        <h4 className="font-semibold">{job.title}</h4>
                        <p className="text-sm text-muted-foreground mt-1">{job.company} • {job.location} • {job.salary}</p>
                        <div className="flex gap-2 mt-2">
                          {job.matching_skills.slice(0, 3).map((skill) => (
                            <SkillBadge key={skill} skill={skill} variant="success" />
                          ))}
                          {job.requiredSkills.filter(skill => !job.matching_skills.includes(skill)).slice(0, 2).map((skill) => (
                            <SkillBadge key={skill} skill={skill} variant="warning" />
                          ))}
                        </div>
                      </div>
                    )) || (
                      <div className="p-4 rounded-lg border bg-card/50">
                        <p className="text-sm text-muted-foreground">No job recommendations available</p>
                      </div>
                    )}
                  </div>
                  <Button className="w-full">View All Jobs</Button>
                </TabsContent>
                <TabsContent value="gaps" className="space-y-4 mt-4">
                  <p className="text-sm text-muted-foreground">
                    Skills you need to develop based on job market analysis:
                  </p>
                  <div className="space-y-3">
                    {analysisData?.skill_gap_analysis.missing_skills.slice(0, 5).map((skill, index) => (
                      <div key={skill} className="flex items-center justify-between p-3 rounded-lg border bg-card/50">
                        <span className="font-medium capitalize">{skill}</span>
                        <SkillBadge 
                          skill={index === 0 ? "High Priority" : index < 3 ? "Medium Priority" : "Low Priority"} 
                          variant={index === 0 ? "destructive" : "warning"} 
                        />
                      </div>
                    )) || (
                      <div className="p-3 rounded-lg border bg-card/50">
                        <p className="text-sm text-muted-foreground">No skill gaps identified! Great job matching.</p>
                      </div>
                    )}
                  </div>
                </TabsContent>
                <TabsContent value="courses" className="space-y-4 mt-4">
                  <div className="space-y-3">
                    {analysisData?.course_recommendations.popular_courses.slice(0, 3).map((course) => (
                      <div key={course.title} className="p-4 rounded-lg border bg-card/50 hover:bg-card transition-colors">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-semibold">{course.title}</h4>
                            <p className="text-sm text-muted-foreground mt-1">{course.provider} • ⭐ {course.rating} • {course.enrollment}</p>
                          </div>
                          <SkillBadge skill={course.category} variant="default" />
                        </div>
                      </div>
                    )) || (
                      <div className="p-4 rounded-lg border bg-card/50">
                        <p className="text-sm text-muted-foreground">No course recommendations available</p>
                      </div>
                    )}
                  </div>
                  <Button className="w-full">Browse All Courses</Button>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}


