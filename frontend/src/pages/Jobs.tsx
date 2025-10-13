import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import SkillBadge from "@/components/SkillBadge";
import { Search, MapPin, DollarSign, Briefcase, Bookmark, ExternalLink } from "lucide-react";

export default function Jobs() {
  const [searchQuery, setSearchQuery] = useState("");

  const mockJobs = [
    {
      id: 1,
      title: "Senior Machine Learning Engineer",
      company: "TechVision AI",
      location: "San Francisco, CA (Remote)",
      salary: "$150k - $200k",
      matchScore: 92,
      skills: ["Python", "TensorFlow", "PyTorch", "ML", "Deep Learning"],
      posted: "2 days ago",
    },
    {
      id: 2,
      title: "Full Stack Developer",
      company: "CloudScale Solutions",
      location: "New York, NY (Hybrid)",
      salary: "$120k - $160k",
      matchScore: 85,
      skills: ["React", "Node.js", "TypeScript", "AWS", "Docker"],
      posted: "1 week ago",
    },
    {
      id: 3,
      title: "Data Science Lead",
      company: "DataDriven Inc",
      location: "Austin, TX (Remote)",
      salary: "$140k - $180k",
      matchScore: 78,
      skills: ["Python", "SQL", "Machine Learning", "Statistics", "Data Visualization"],
      posted: "3 days ago",
    },
    {
      id: 4,
      title: "AI Research Scientist",
      company: "Innovation Labs",
      location: "Boston, MA",
      salary: "$160k - $220k",
      matchScore: 88,
      skills: ["Python", "Research", "Deep Learning", "NLP", "Computer Vision"],
      posted: "5 days ago",
    },
  ];

  const getMatchColor = (score: number) => {
    if (score >= 85) return "success";
    if (score >= 70) return "warning";
    return "destructive";
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="space-y-4">
        <h1 className="text-4xl font-bold">Job Recommendations</h1>
        <p className="text-lg text-muted-foreground">
          Personalized job matches based on your skills and experience
        </p>
      </div>

      {/* Search and Filters */}
      <Card className="glass-card border shadow-lg">
        <CardContent className="pt-6">
          <div className="flex gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-5 h-5" />
              <Input
                placeholder="Search jobs by title, skills, or company..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <Button className="shadow-md">
              <Search className="w-4 h-4 mr-2" />
              Search
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Job Listings */}
      <div className="space-y-4">
        {mockJobs.map((job) => (
          <Card key={job.id} className="glass-card border shadow-md hover:shadow-lg transition-all group">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <CardTitle className="text-xl group-hover:text-primary transition-colors">
                      {job.title}
                    </CardTitle>
                    <Badge
                      variant={getMatchColor(job.matchScore) as any}
                      className="px-3 py-1"
                    >
                      {job.matchScore}% Match
                    </Badge>
                  </div>
                  <CardDescription className="flex flex-wrap items-center gap-4 text-base">
                    <span className="flex items-center gap-1">
                      <Briefcase className="w-4 h-4" />
                      {job.company}
                    </span>
                    <span className="flex items-center gap-1">
                      <MapPin className="w-4 h-4" />
                      {job.location}
                    </span>
                    <span className="flex items-center gap-1">
                      <DollarSign className="w-4 h-4" />
                      {job.salary}
                    </span>
                  </CardDescription>
                </div>
                <Button variant="ghost" size="icon" className="hover:text-primary">
                  <Bookmark className="w-5 h-5" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-2">
                {job.skills.map((skill) => (
                  <SkillBadge key={skill} skill={skill} variant="default" />
                ))}
              </div>
              <div className="flex items-center justify-between pt-4 border-t">
                <span className="text-sm text-muted-foreground">Posted {job.posted}</span>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm">
                    <Bookmark className="w-4 h-4 mr-2" />
                    Save
                  </Button>
                  <Button size="sm" className="shadow-md">
                    Apply Now
                    <ExternalLink className="w-4 h-4 ml-2" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Load More */}
      <div className="flex justify-center pt-4">
        <Button variant="outline" size="lg">
          Load More Jobs
        </Button>
      </div>
    </div>
  );
}
