import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import SkillBadge from "@/components/SkillBadge";
import { Search, ExternalLink, Clock, Star, DollarSign, BookOpen } from "lucide-react";

export default function Courses() {
  const [searchQuery, setSearchQuery] = useState("");

  const mockCourses = [
    {
      id: 1,
      title: "Complete Machine Learning Masterclass",
      platform: "Coursera",
      instructor: "Dr. Sarah Chen",
      rating: 4.9,
      students: "125k",
      duration: "40 hours",
      price: "Free",
      level: "Beginner to Advanced",
      skills: ["Machine Learning", "Python", "Data Science"],
    },
    {
      id: 2,
      title: "AWS Certified Solutions Architect",
      platform: "Udemy",
      instructor: "John Smith",
      rating: 4.7,
      students: "89k",
      duration: "30 hours",
      price: "$49.99",
      level: "Intermediate",
      skills: ["AWS", "Cloud Computing", "DevOps"],
    },
    {
      id: 3,
      title: "Advanced TypeScript Programming",
      platform: "Pluralsight",
      instructor: "Emily Rodriguez",
      rating: 4.8,
      students: "56k",
      duration: "25 hours",
      price: "$29.99",
      level: "Advanced",
      skills: ["TypeScript", "JavaScript", "Web Development"],
    },
    {
      id: 4,
      title: "Docker and Kubernetes Bootcamp",
      platform: "Udemy",
      instructor: "Michael Lee",
      rating: 4.9,
      students: "142k",
      duration: "35 hours",
      price: "$59.99",
      level: "Intermediate",
      skills: ["Docker", "Kubernetes", "DevOps"],
    },
  ];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="space-y-4">
        <h1 className="text-4xl font-bold">Course Recommendations</h1>
        <p className="text-lg text-muted-foreground">
          Personalized learning paths to develop your skills and advance your career
        </p>
      </div>

      {/* Search */}
      <Card className="glass-card border shadow-lg">
        <CardContent className="pt-6">
          <div className="flex gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-5 h-5" />
              <Input
                placeholder="Search courses by skill or topic..."
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

      {/* Filters */}
      <Tabs defaultValue="all" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="all">All Courses</TabsTrigger>
          <TabsTrigger value="free">Free</TabsTrigger>
          <TabsTrigger value="paid">Paid</TabsTrigger>
          <TabsTrigger value="beginner">Beginner</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="space-y-4 mt-6">
          <div className="grid gap-6 md:grid-cols-2">
            {mockCourses.map((course) => (
              <Card key={course.id} className="glass-card border shadow-md hover:shadow-lg transition-all group">
                <CardHeader>
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <CardTitle className="text-lg group-hover:text-primary transition-colors leading-tight">
                        {course.title}
                      </CardTitle>
                      <CardDescription className="mt-2">
                        <span className="flex items-center gap-2 text-sm">
                          <BookOpen className="w-4 h-4" />
                          {course.platform} • {course.instructor}
                        </span>
                      </CardDescription>
                    </div>
                    <div className="text-right">
                      <div className="flex items-center gap-1 text-sm font-semibold mb-1">
                        <Star className="w-4 h-4 fill-warning text-warning" />
                        {course.rating}
                      </div>
                      <p className="text-xs text-muted-foreground">{course.students}</p>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex flex-wrap gap-2">
                    {course.skills.map((skill) => (
                      <SkillBadge key={skill} skill={skill} variant="default" />
                    ))}
                  </div>

                  <div className="flex items-center justify-between text-sm text-muted-foreground">
                    <div className="flex items-center gap-4">
                      <span className="flex items-center gap-1">
                        <Clock className="w-4 h-4" />
                        {course.duration}
                      </span>
                      <SkillBadge skill={course.level} variant="default" />
                    </div>
                    <span className="flex items-center gap-1 font-semibold text-primary">
                      <DollarSign className="w-4 h-4" />
                      {course.price}
                    </span>
                  </div>

                  <div className="pt-4 border-t">
                    <Button className="w-full shadow-md group-hover:shadow-lg">
                      Enroll Now
                      <ExternalLink className="w-4 h-4 ml-2" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="free" className="mt-6">
          <p className="text-center text-muted-foreground py-12">
            Showing free courses only
          </p>
        </TabsContent>

        <TabsContent value="paid" className="mt-6">
          <p className="text-center text-muted-foreground py-12">
            Showing paid courses only
          </p>
        </TabsContent>

        <TabsContent value="beginner" className="mt-6">
          <p className="text-center text-muted-foreground py-12">
            Showing beginner-friendly courses
          </p>
        </TabsContent>
      </Tabs>

      {/* Learning Path */}
      <Card className="glass-card border shadow-lg">
        <CardHeader>
          <CardTitle>Your Personalized Learning Path</CardTitle>
          <CardDescription>Based on your skill gaps and career goals</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative space-y-6">
            {/* Timeline */}
            <div className="absolute left-8 top-8 bottom-8 w-0.5 bg-gradient-to-b from-primary via-accent to-success" />

            {["Foundation", "Intermediate", "Advanced"].map((level, idx) => (
              <div key={level} className="flex gap-6 relative">
                <div className="w-16 h-16 rounded-full gradient-primary flex items-center justify-center font-bold text-white shadow-glow z-10">
                  {idx + 1}
                </div>
                <div className="flex-1 pb-8">
                  <h4 className="font-semibold text-lg mb-2">{level} Level</h4>
                  <p className="text-sm text-muted-foreground mb-3">
                    Complete 2-3 courses to master {level.toLowerCase()} concepts
                  </p>
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline">
                      View Courses
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
